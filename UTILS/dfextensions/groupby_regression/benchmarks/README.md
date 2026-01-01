# GroupBy Regression Benchmarks

Performance benchmarks for the `groupby_regression` module.

## Overview

Three benchmark categories:

| Benchmark | Purpose | Measures |
|-----------|---------|----------|
| `bench_v5.py` | High-level API | End-to-end throughput with joblib parallelism |
| `bench_groupby_regression_kernels.py` | Low-level Numba kernels | Kernel speedup vs NumPy baseline |
| `bench_groupby_regression_memory.py` | Memory stability | RSS drift and fragmentation |

## Quick Start

```bash
# Run all benchmarks (quick suite)
source ./benchmarks/run_benchmarks.sh

# Or individually:
python benchmarks/bench_groupby_regression_kernels.py --quick
python benchmarks/bench_groupby_regression_memory.py --quick
python benchmarks/bench_v5.py --quick
```

## Generating Review Artifacts

To generate log and JSON files for code review:

```bash
cd groupby_regression
source ./benchmarks/run_benchmarks.sh --review
```

This creates a timestamped directory with:

**Standalone benchmarks:**
- `kernel_results.log` — Console output from kernel benchmarks
- `kernel_results.json` — Structured results for kernel benchmarks
- `memory_results.log` — Console output from memory benchmarks  
- `memory_results.json` — Structured results for memory benchmarks

**BF integration:**
- `bf_results.log` — Console output from BF runner
- `bf_results.json` — Structured BF output (proves discovery + schema mapping)

Submit all 6 files with your review request.

## Kernel Benchmarks

### What It Measures

- **Numba vs NumPy speedup**: How much faster are the Numba kernels?
- **Single-fit performance**: `fit_groups_single_numba()` throughput
- **Multi-fit performance**: `fit_groups_multi_numba()` batch efficiency
- **Correctness**: Results match NumPy reference implementation

### Scenarios

| ID | Groups | Rows/Group | Features | Description |
|----|--------|------------|----------|-------------|
| K1 | 500 | 20 | 2 | Quick (CI) |
| K2 | 1,000 | 20 | 2 | Small |
| K3 | 5,000 | 30 | 2 | Medium |
| K4 | 10,000 | 20 | 2 | Large |
| K5 | 5,000 | 50 | 4 | Wide (more features) |
| K6 | 100,000 | 20 | 2 | Stress test |

### Gates

| Gate | Threshold | Rationale |
|------|-----------|-----------|
| Correctness | max_error < 0.2 | Numerical agreement with NumPy |
| Numba speedup | ≥5× vs NumPy | Minimum acceptable acceleration |

### Usage

```bash
# Quick validation (K1-K2 only)
python benchmarks/bench_groupby_regression_kernels.py --quick

# Full benchmark (all scenarios)
python benchmarks/bench_groupby_regression_kernels.py

# Save JSON results
python benchmarks/bench_groupby_regression_kernels.py --json results.json
```

### Output Example

```
============================================================
Scenario: Large: 10K groups
============================================================
  Groups: 10,000, Rows/group: 20, Features: 2
  NumPy: 124.40 ms (80,386 groups/sec)
  ✓ Correctness: PASS (max error: 9.69e-02)
  Numba: 2.37 ms → 52.4× vs NumPy
```

## Memory Benchmarks

### What It Measures

- **RSS drift**: Memory growth over repeated kernel calls
- **RSS CV (coefficient of variation)**: Memory stability/fragmentation
- **Peak RSS**: Maximum memory usage

### Why This Matters

Memory leaks or fragmentation in batch farm environments (2-4 GB/core limit) can cause job failures. These benchmarks verify:
1. No memory accumulation across iterations
2. Stable allocation patterns (low CV)

### Scenarios

| ID | Groups | Rows/Group | Features | Description |
|----|--------|------------|----------|-------------|
| M1 | 500 | 20 | 2 | Small |
| M2 | 2,000 | 30 | 2 | Medium |
| M3 | 5,000 | 20 | 2 | Large |
| M4 | 1,000 | 50 | 8 | Wide |

### Gates

| Gate | Threshold | Rationale |
|------|-----------|-----------|
| RSS drift | < 5% | No significant memory growth |
| RSS CV | < 0.1 | Stable allocation pattern |

### Usage

```bash
# Quick check
python benchmarks/bench_groupby_regression_memory.py --quick

# Full benchmark (50 iterations)
python benchmarks/bench_groupby_regression_memory.py

# Custom iterations
python benchmarks/bench_groupby_regression_memory.py --iterations 100

# Save results
python benchmarks/bench_groupby_regression_memory.py --json results.json
```

### Output Example

```
================================================================================
MEMORY BENCHMARK SUITE
================================================================================

Small: drift=+0.0%, CV=0.000
Medium: drift=+0.0%, CV=0.000
Large: drift=+0.01%, CV=0.000

================================================================================
ALL MEMORY GATES PASSED ✓
```

## V5 API Benchmarks

### What It Measures

- **End-to-end throughput**: Rows processed per second
- **Parallel scaling**: Performance across n_jobs values
- **Real-world scenarios**: DataFrame operations with metadata

### Scenarios

| ID | Rows | Groups | Description |
|----|------|--------|-------------|
| S1 | 50K | 1K | Quick |
| S2 | 100K | 1K | Small |
| S3 | 200K | 1K | Medium |
| S4 | 400K | 5K | Large |
| S5 | 25M | 25K | Stress (slow) |

### Usage

```bash
# Quick suite (S1-S2)
python benchmarks/bench_v5.py --quick

# Full suite (S1-S5)
python benchmarks/bench_v5.py
```

## Benchmark Framework Integration

These benchmarks integrate with the Benchmark Framework (BF) for automated regression detection:

```bash
# Via BF runner (discovers all benchmarks)
python -m dfextensions.benchmarks.runner --subproject groupby_regression

# Quick suite
python -m dfextensions.benchmarks.runner --subproject groupby_regression --suite quick

# Release suite (includes stress tests)
python -m dfextensions.benchmarks.runner --subproject groupby_regression --suite release
```

### BF Discovery

The runner discovers these benchmarks:
- `v5_batch_fit` from bench_v5.py
- `kernel_single_fit` from bench_groupby_regression_kernels.py
- `kernel_multi_fit` from bench_groupby_regression_kernels.py
- `memory_rss_tracking` from bench_groupby_regression_memory.py

### Notes on BF Integration

- Kernel/memory benchmarks **do not use n_jobs** (they benchmark single-threaded Numba kernels)
- V5 benchmarks **do use n_jobs** (they benchmark joblib parallelism)
- The runner handles this automatically

## JSON Output Schema

### Kernel Benchmark

```json
{
  "timestamp": "2026-01-01T11:17:59",
  "results": [
    {
      "name": "Large: 10K groups",
      "n_groups": 10000,
      "rows_per_group": 20,
      "n_feat": 2,
      "numpy_time_ms": 124.4,
      "single_time_ms": 2.37,
      "numba_vs_numpy_speedup": 52.4,
      "single_correctness": true,
      "single_max_error": 0.097
    }
  ],
  "gates": {"all_pass": true}
}
```

### Memory Benchmark

```json
{
  "results": [
    {
      "name": "Large",
      "n_groups": 10000,
      "n_iterations": 50,
      "rss_drift_pct": 0.01,
      "rss_cv": 0.0,
      "gates": {"rss_drift_ok": true, "rss_cv_ok": true}
    }
  ],
  "all_pass": true
}
```

## Troubleshooting

### "Gate FAIL: Numba vs NumPy < 5×"

This can happen when:
1. **First run**: JIT compilation overhead. Re-run the benchmark.
2. **Small data**: Overhead dominates. Use larger scenarios.
3. **CPU throttling**: Laptop in power-save mode. Check CPU frequency.

### Memory drift > 5%

Check for:
1. Global state accumulation
2. Unbounded caches
3. Memory fragmentation (increase iterations to confirm pattern)

### Slow S5 scenario

S5 (25M rows) takes ~8-10 seconds per configuration. Use `--quick` for CI:

```bash
python benchmarks/bench_v5.py --quick  # S1-S2 only
```

## File Manifest

```
groupby_regression/benchmarks/
├── README.md                              # This file
├── run_benchmarks.sh                      # Convenience script
├── bench_v5.py                            # High-level API benchmarks
├── bench_groupby_regression_kernels.py    # Kernel performance
└── bench_groupby_regression_memory.py     # Memory stability
```

## Version History

| Phase | Date | Changes |
|-------|------|---------|
| 12.14a.GB | 2025-12-31 | Initial kernel + memory benchmarks |
| 12.14b.GB | 2026-01-01 | BF integration, scenarios K1-K6/M1-M4 |
