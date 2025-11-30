# AliasDataFrame Benchmarks

Performance benchmarks for AliasDataFrame operations.

## Quick Start

```bash
# Run all benchmarks (auto-generates synthetic data if needed)
./run_benchmark.sh

# Quick mode for CI
./run_benchmark.sh --quick

# Synthetic only (no ROOT file)
./run_benchmark.sh --synthetic-only
```

## New Features

### Full Analysis Mode

Run complete benchmark with profiling, baseline comparison, and history archiving:

```bash
./run_benchmark.sh --full
```

This enables:
- Profiler output (`.prof` and `.txt` files)
- Baseline comparison
- Automatic history archiving with git info

### Profiler Output

Generate detailed profiler output for performance analysis:

```bash
./run_benchmark.sh --profile

# Or combined with full analysis
./run_benchmark.sh --full
```

Profile files are saved to `results/profiles/` with naming:
```
bench_<component>_<scenario>_<timestamp>_<commit>.prof
```

Analyze profiles with standard Python tools:
```python
import pstats
p = pstats.Stats('results/profiles/bench_materialize_safe_20251130_164906_18caba76.prof')
p.sort_stats('cumulative').print_stats(20)

# Or use snakeviz for visualization
# pip install snakeviz
# snakeviz results/profiles/bench_materialize_safe_20251130_164906_18caba76.prof
```

### History and Comparison

Every benchmark run is archived to `results/history/` with git information.

**Compare two runs:**
```bash
# Compare specific files
python baseline_utils.py diff results/history/benchmark_*_f9df9cf.json results/history/benchmark_*_18caba7.json

# Supports glob patterns
python baseline_utils.py diff 'results/history/*f9df9cf*' 'results/history/*18caba7*'

# With strict mode (exit code 1 on regression)
python baseline_utils.py diff file_a.json file_b.json --strict
```

### History Analysis

Load history into pandas DataFrames for custom analysis:

```python
from history_analysis import load_history_long, load_history_wide

# Long format (one row per metric) - good for filtering
df_long = load_history_long('results/history/')
df_long[df_long['metric'] == 'direct_vs_safe_speedup']

# Wide format (one row per run) - good for correlation
df_wide = load_history_wide('results/history/')
df_wide[['commit', 'materialize_aliases_time_s', 'materialize_aliases_direct_vs_safe_speedup']]

# Time series of specific metric
from history_analysis import get_metric_history
ts = get_metric_history(df_long, 'materialize_aliases', 'direct_vs_safe_speedup')
```

**CLI commands:**
```bash
# List available metrics
python history_analysis.py list results/history/

# Show recent runs
python history_analysis.py show results/history/ --last 10

# Show specific metric
python history_analysis.py show results/history/ --metric direct_vs_safe_speedup

# Export for external tools
python history_analysis.py export results/history/ --format wide -o history.csv
```

## Overview

| Script | Purpose | Data Required |
|--------|---------|---------------|
| `run_benchmark.sh` | **Main entry point** - runs all benchmarks | Auto-generates |
| `benchmark_performance.py` | Core operations timing | Synthetic (built-in) |
| `benchmark_materialize_aliases.py` | **Alias DAG + subframe joins** | Synthetic (built-in) |
| `benchmark_read_tree.py` | ROOT file read tests | ROOT file |
| `benchmark_subframe.py` | Subframe join tests | ROOT file |
| `benchmark_parallel.py` | Worker scaling analysis | ROOT file |
| `generate_synthetic_data.py` | Create test ROOT file | None |
| `diagnose_read_performance.py` | **Diagnose slowdowns** | ROOT file |

## run_benchmark.sh (Main Entry Point)

Runs ALL benchmarks with pytest-style output:

```
==================== BENCHMARK SESSION STARTS ====================
Timestamp:  2025-11-28T14:40:00
Host:       your-machine
Python:     Python 3.9.6

✓ benchmark_performance.py          PASSED (1.98s)
✓ benchmark_materialize_aliases.py  PASSED (3.05s)
✓ benchmark_read_tree.py            PASSED (0.73s)
✓ benchmark_subframe.py             PASSED (2.52s)
✓ benchmark_parallel.py             PASSED (6.23s)

==================== 5 passed in 14.51s ====================
```

### Options

```bash
./run_benchmark.sh                 # Full benchmarks
./run_benchmark.sh --quick         # Quick mode (smaller data)
./run_benchmark.sh --synthetic-only # Only synthetic (no ROOT)
./run_benchmark.sh --generate-data  # Force regenerate test data
./run_benchmark.sh --strict        # Exit 1 on failure
./run_benchmark.sh --verbose       # Show detailed output
./run_benchmark.sh --save-baseline # Save results as new baseline
./run_benchmark.sh --compare-baseline # Compare against baseline
```

### Regression Detection

```bash
# Establish baseline
./run_benchmark.sh --save-baseline

# Compare future runs against baseline
./run_benchmark.sh --compare-baseline

# Fail CI if regression detected (>20% slower)
./run_benchmark.sh --compare-baseline --strict
```

## benchmark_performance.py

Tests core AliasDataFrame operations with synthetic data.

### Operations Tested

| Operation | Description | Threshold (1M rows) |
|-----------|-------------|---------------------|
| `create_adf` | Create AliasDataFrame from DataFrame | < 0.5s |
| `add_aliases` | Add 50 expression aliases | < 1.0s |
| `validate_schema` | Validate schema consistency | < 1.0s |
| `materialize` | Materialize one alias to column | < 0.5s |
| `compress` | Compress one column | < 2.0s |
| `export_schema` | Export definition schema | < 0.5s |

### Usage

```bash
# Full benchmark (1M rows, 50 aliases)
python benchmark_performance.py

# Quick mode (100k rows, 10 aliases) - for CI
python benchmark_performance.py --quick

# Custom row count
python benchmark_performance.py --rows 500000

# Export results to JSON
python benchmark_performance.py --json results.json

# Save current results as baseline
python benchmark_performance.py --update-baselines

# Minimal output
python benchmark_performance.py --quiet
```

### Output

```
======================================================================
SYNTHETIC PERFORMANCE BENCHMARK
======================================================================
Rows:      1,000,000
Aliases:   50
...

--- 2. AliasDataFrame creation ---
  ✓ Time: 0.150s (threshold: 0.50s)
    Memory: 45.2 MB
...

======================================================================
SUMMARY
======================================================================
Operation            Time (s)     Threshold    Memory (MB)  Status  
----------------------------------------------------------------------
create_adf           0.150        0.50         45.2         ✓ PASS  
add_aliases          0.320        1.00         12.1         ✓ PASS  
...

✓ ALL BENCHMARKS PASSED
======================================================================
```

### Baselines

After running `--update-baselines`, future runs will compare against saved baselines:

```bash
# First run: establish baseline
python benchmark_performance.py --update-baselines

# Future runs: compare to baseline
python benchmark_performance.py
```

Results show regression warnings if current run is >2x slower than baseline.

## benchmark_materialize_aliases.py

Tests `materialize_aliases()` performance with realistic physics data scenarios including subframe joins.

### Purpose

- Measure alias DAG materialization performance
- Compare `fill_mode='safe'` vs `fill_mode='direct'`
- Quantify subframe join overhead
- Detect regressions in batch materialization (BUG-2025-11-27-002 fix)

### Data Model (ITSTPC-like)

| Dataset | Rows | Columns | Description |
|---------|------|---------|-------------|
| Main DataFrame | 500k | 8 | drift25, side, row, r, phi, y2x, dyC1, dzC1 |
| Subframe | 1,288 | 11 | Calibration table with 3-column join key |

The data simulates ALICE TPC calibration with ~15% missing keys (rows 161-190 have no calibration data).

### Alias DAG Structure

26 aliases organized in 4 layers:

| Layer | Aliases | Description |
|-------|---------|-------------|
| A | rrel, cosPhi, sinPhi, y2x2 | Geometry calculations |
| B | dyC1_SC, dzC1_SC | Subframe projections (joins) |
| C | dyC1_SC_combined | Intermediate corrections |
| D | dyC2, dzC2 | Final calibrated residuals |

### Scenarios

| Scenario | Aliases | Description |
|----------|---------|-------------|
| `simple` | 11 | No subframe joins (baseline) |
| `safe` | 26 | Full NaN/Inf checking with subframe |
| `direct` | 26 | Skip NaN/Inf checks (faster) |

### Usage

```bash
# Full benchmark (500k rows)
python benchmark_materialize_aliases.py

# Quick mode (100k rows) - for CI
python benchmark_materialize_aliases.py --quick

# Export results to JSON
python benchmark_materialize_aliases.py --json results.json

# Minimal output
python benchmark_materialize_aliases.py --quiet
```

### Output

```
============================================================
MATERIALIZE_ALIASES BENCHMARK
============================================================
Rows:      500,000
Hostname:  your-machine
Timestamp: 2025-11-28T14:40:14

Generating synthetic data...
  Main DataFrame: 500,000 rows × 8 cols
  Subframe: 1,288 rows × 11 cols
  Expected missing: 15.2% (row > 160)

--- Scenario 1: Simple (no subframe) ---
  Aliases defined: 11
  Targets: ['simple_result']
  Time: 0.019s
  Rows/sec: 26,751,976
  Peak memory: 82.2 MB

--- Scenario: Subframe (Safe) ---
  Aliases defined: 26
  Targets: ['dyC2', 'dzC2']
  Fill mode: safe
  Time: 0.767s
  Rows/sec: 651,941
  Peak memory: 196.7 MB
  Missing keys: 15.2%

--- Scenario: Subframe (Direct) ---
  Aliases defined: 26
  Fill mode: direct
  Time: 0.706s
  Rows/sec: 707,936
  Peak memory: 196.6 MB

============================================================
SUMMARY
============================================================

Scenario        Time (s)     Rows/sec        Aliases   
------------------------------------------------------------
simple          0.019        26,751,976      11        
safe            0.767        651,941         26        
direct          0.706        707,936         26        
------------------------------------------------------------
Total           1.492       

------------------------------------------------------------
SPEEDUP METRICS
------------------------------------------------------------
  direct vs safe:   1.09x (faster)
  safe vs simple:   41x (subframe overhead)
============================================================
```

### Key Metrics

| Metric | Description | Typical Value |
|--------|-------------|---------------|
| `direct_vs_safe_speedup` | Speed gain from skipping NaN checks | ~1.05-1.10x |
| `safe_vs_simple_ratio` | Subframe join overhead | ~40-50x |
| `missing_pct` | Percentage of missing join keys | 15.2% |

### Roofline Analysis (Efficiency Metrics)

The benchmark measures theoretical performance limits and calculates efficiency:

| Metric | Description |
|--------|-------------|
| `memory_bandwidth` | Raw numpy.copy() speed (absolute floor) |
| `numpy_indexing_join` | NumPy advanced indexing (ideal join target) |
| `efficiency` | `theoretical_time / actual_time` (higher = better, max = 100%) |

**Interpreting efficiency:**
- **>50%**: Near optimal, limited optimization potential
- **10-50%**: Room for optimization
- **<10%**: Significant framework overhead, investigate

Example output:
```
============================================================
EFFICIENCY (vs Theoretical Limits)
============================================================
Memory bandwidth: 15.2 GB/s
NumPy indexing:   0.0080s (8 cols × 1,000,000 rows)

Scenario         Time      Limit   Efficiency
----------------------------------------------
simple          0.019s    0.0020s       10.5%
safe            0.767s    0.0080s        1.0%
direct          0.706s    0.0080s        1.1%
----------------------------------------------

Interpretation:
  >50%  : Near optimal
  10-50%: Room for optimization
  <10%  : Significant overhead (investigate)
```

The efficiency values show how close we are to theoretical limits. Low efficiency in safe/direct modes indicates the join overhead dominates, which is the target for Phase 3 optimization.

### Interpreting Results

**Subframe Overhead (safe_vs_simple):**
- 30-50x is normal (join operations are expensive)
- >100x may indicate inefficient join strategy
- Use for comparison across code changes, not absolute benchmarking

**Direct vs Safe Speedup:**
- 1.05-1.15x expected (NaN/Inf checks have cost)
- <1.0x indicates regression in direct mode
- Use `fill_mode='direct'` when input data is pre-validated

## benchmark_parallel.py

Tests `read_tree` performance with different `num_workers` values.

### Purpose

- Find optimal worker count for your system
- Detect parallel reading issues (hangs, timeouts)
- Identify resource contention on shared servers

### Usage

```bash
# Basic run
python benchmark_parallel.py data.root

# Test more workers
python benchmark_parallel.py data.root --max-workers 16

# Specific worker counts
python benchmark_parallel.py data.root --workers "1,2,4,8,12"

# Adjust timeout and repeats
python benchmark_parallel.py data.root --timeout 30 --repeats 5

# Export results
python benchmark_parallel.py data.root --json parallel_results.json
```

### Output

```
======================================================================
PARALLEL READ BENCHMARK
======================================================================
File:      /path/to/data.root
Workers:   [1, 2, 4, 8]
Repeats:   3
Timeout:   60s

Testing num_workers=1...
  Run 1: 5.23s (1,000,000 rows, 125.4 MB)
  Run 2: 5.18s (1,000,000 rows, 125.4 MB)
  Run 3: 5.21s (1,000,000 rows, 125.4 MB)
  → Mean: 5.21s ± 0.02s

Testing num_workers=4...
  Run 1: 2.65s (1,000,000 rows, 125.4 MB)
  ...

======================================================================
SUMMARY
======================================================================
Workers    Mean (s)     Std (s)      Min (s)      Failures  
----------------------------------------------------------------------
1          5.21         0.02         5.18         0/3       
2          3.45         0.05         3.40         0/3       
4          2.65         0.03         2.62         0/3        *
8          2.71         0.15         2.58         0/3       

* Optimal: num_workers=4

======================================================================
ISSUES DETECTED
======================================================================
  ✓ No issues detected

======================================================================
RECOMMENDATIONS
======================================================================
  → Recommended: num_workers=4 (2.65s average)
======================================================================
```

### Platform Note

⚠️ **Timeout detection requires POSIX (Linux/macOS).**

On Windows, timeout is disabled and hangs will block indefinitely. 
Use `--timeout 0` to acknowledge this and run without timeout protection.

## benchmark_read_tree.py

Comprehensive benchmark comparing different ROOT file reading strategies.

### Purpose

- Compare uproot reading methods (one-shot, branch-by-branch, threaded)
- Measure impact of dtype conversion (float64 → float16)
- Find optimal reading strategy for your data/system
- Identify I/O vs CPU bottlenecks

### Tests Performed

| Test | Description |
|------|-------------|
| Test 1 | Uproot one-shot `arrays(library='np')` |
| Test 2 | Branch-by-branch read (no conversion) |
| Test 3 | Branch-by-branch with dtype conversion |
| Test 4 | **AliasDataFrame.read_tree** (optimized) |
| Test 5 | Uproot one-shot with ThreadPoolExecutor |
| Test 6 | Uproot `arrays(library='pd')` (baseline) |
| Test 7 | Branch-by-branch threaded (no conversion) |
| Test 8 | Branch-by-branch threaded with conversion |

### Usage

```bash
# Basic run
python benchmark_read_tree.py data.root tree

# Limit entries
python benchmark_read_tree.py data.root tree 1000000

# Entry range
python benchmark_read_tree.py data.root tree --entry-start=0 --entry-stop=500000

# Custom workers
python benchmark_read_tree.py data.root tree --workers 4
```

### Output

```
======================================================================
AliasDataFrame Read Performance Benchmark
======================================================================
File: data.root
Tree: tree
Range: 0 : 1000000
Workers: 8
======================================================================

[6] Uproot arrays(library='pd') - original method
    Time: 12.5s
    Peak memory: 450 MB
    DataFrame: 380 MB

[1] Uproot arrays(library='np') - one-shot
    Time: 8.2s
    Peak memory: 420 MB
    ...

======================================================================
SUMMARY
======================================================================
Test            Time (s)     Peak (MB)    Final (MB)  
---------------------------------------------------
6-pd-direct     12.5         450          380         
1-np-oneshot    8.2          420          380         
...

======================================================================
ANALYSIS
======================================================================
numpy vs pandas direct: +52% (faster)
...
```

## diagnose_read_performance.py

Diagnostic tool to identify root causes of read performance issues.

### Tests Available

| Test | What it measures | Identifies |
|------|-----------------|------------|
| `workers` | Scaling with num_workers | Thread contention |
| `cache` | Cold vs warm reads | Filesystem caching |
| `local` | Network vs /tmp storage | I/O bandwidth bottleneck |
| `overhead` | Raw uproot vs AliasDataFrame | ADF processing overhead |

### Usage

```bash
# Run all diagnostic tests
python diagnose_read_performance.py data.root

# Run specific test
python diagnose_read_performance.py data.root --test workers
python diagnose_read_performance.py data.root --test cache
python diagnose_read_performance.py data.root --test local
python diagnose_read_performance.py data.root --test overhead

# Quick test with limited entries
python diagnose_read_performance.py data.root --entries 100000

# Custom worker counts
python diagnose_read_performance.py data.root --workers "1,2,4,8,12,16,32"

# Export results to JSON
python diagnose_read_performance.py data.root --json results.json
```

### Output Example

```
============================================================
ALIASDATAFRAME READ PERFORMANCE DIAGNOSTIC
============================================================
File:      data.root
Size:      254.6 MB
Tree:      tree

Host:      lxbk1130
Platform:  Linux-4.18.0-x86_64
CPUs:      256
FS Type:   lustre

============================================================
TEST: WORKER SCALING
============================================================
Purpose: Find optimal worker count, detect thread contention

  workers= 1:    4.50s  speedup=(baseline)  (12,630,498 rows)
  workers= 2:    1.83s  speedup=   2.46x    (12,630,498 rows)
  workers= 4:    1.12s  speedup=   4.00x    (12,630,498 rows)
  workers= 8:    0.83s  speedup=   5.45x    (12,630,498 rows)
  workers=16:    0.81s  speedup=   5.56x    (12,630,498 rows)

Analysis:
  Best:  workers=8 (0.83s)
  Worst: workers=1 (4.50s)

============================================================
TEST: COLD VS WARM CACHE
============================================================
Purpose: Measure filesystem caching effect (5 sequential reads)

  Read 1 (COLD  ):    1.18s  (12,630,498 rows)
  Read 2 (WARM-1):    1.16s  (12,630,498 rows)
  Read 3 (WARM-2):    1.14s  (12,630,498 rows)
  Read 4 (WARM-3):    1.14s  (12,630,498 rows)
  Read 5 (WARM-4):    1.12s  (12,630,498 rows)

Analysis:
  Cold read:     1.18s
  Warm average:  1.14s
  Cache benefit: 3%

✓ Storage location has minimal impact.

============================================================
TEST: UPROOT VS ALIASDATAFRAME OVERHEAD
============================================================
Purpose: Measure AliasDataFrame processing overhead

  Raw uproot:      2.58s  (12,630,498 rows)
  AliasDataFrame:  1.13s  (workers=4)

Analysis:
  Overhead: -56% (0.44x)

✓ AliasDataFrame is FASTER than raw uproot!
  Threaded branch-by-branch reading provides speedup.

============================================================
RECOMMENDATIONS
============================================================
• Use num_workers=8 for best performance
```

### Interpreting Results

**Worker Scaling:**
- Linear speedup up to N workers → Good parallelization
- Speedup plateaus early → I/O bound or GIL contention
- More workers = slower → Thread contention, reduce workers

**Cache Effect:**
- Cold >> Warm → Filesystem caching significant, pre-warm for batch jobs
- Cold ≈ Warm → No caching issues

**Local vs Network:**
- >2x speedup local → Copy to /tmp before processing
- <1.5x speedup → Network storage is fine

**Overhead:**
- Negative overhead (ADF faster) → Threaded reading working well
- <20% overhead → Normal
- >100% overhead → Check subframes, reduce workers

### JSON Output

Results can be exported to JSON for programmatic analysis:

```json
{
  "system_info": {
    "hostname": "lxbk1130",
    "platform": "Linux-4.18.0-x86_64",
    "cpu_count": 256,
    "filesystem_type": "lustre"
  },
  "workers": {
    "results": [
      {"workers": 1, "time_s": 4.50, "speedup": 1.0},
      {"workers": 4, "time_s": 1.12, "speedup": 4.0},
      {"workers": 8, "time_s": 0.83, "speedup": 5.45}
    ]
  },
  "overhead": {
    "results": {
      "uproot": {"time_s": 2.58},
      "aliasdataframe": {"time_s": 1.13},
      "overhead_pct": -56.2
    }
  }
}
```

## JSON Output Format

All benchmarks support `--json` output for programmatic processing:

```json
{
  "timestamp": "2025-11-28T14:40:00.123456",
  "hostname": "your-machine",
  "python_version": "3.9.6",
  "platform": "Linux-5.4.0-x86_64",
  "rows": 1000000,
  "mode": "full",
  "results": {
    "create_adf": {
      "time_s": 0.15,
      "memory_mb": 45.2,
      "threshold_s": 0.5,
      "passed": true
    }
  },
  "all_passed": true,
  "total_time_s": 5.23,
  "total_memory_mb": 125.4
}
```

## Directory Structure

```
benchmarks/
├── README.md                         # This file
├── run_benchmark.sh                  # Main entry point (pytest-style)
├── generate_synthetic_data.py        # Creates test ROOT file (~5MB)
├── diagnose_read_performance.py      # Diagnostic tool for slowdowns
├── benchmark_performance.py          # Core operations timing
├── benchmark_materialize_aliases.py  # Alias DAG + subframe benchmark
├── benchmark_parallel.py             # Parallel scaling tests
├── benchmark_read_tree.py            # ROOT file read comparison
├── benchmark_subframe.py             # Subframe validation
├── baseline_utils.py                 # Baseline management utilities
├── history_analysis.py               # DataFrame utilities for history analysis (NEW)
├── baselines.json                    # Saved baselines (auto-generated)
├── baseline.json                     # Unified baseline for regression detection
├── synthetic_data.root               # Test data (auto-generated, gitignored)
└── results/                          # Output directory (gitignored)
    ├── history/                      # Archived runs with git info (NEW)
    │   ├── benchmark_20251128_150047_f9df9cf.json
    │   └── benchmark_20251130_164906_18caba76.json
    ├── profiles/                     # Profiler output (NEW)
    │   ├── bench_materialize_safe_20251130_164906_18caba76.prof
    │   ├── bench_materialize_safe_20251130_164906_18caba76.txt
    │   └── ...
    ├── benchmark_*.json              # Detailed results
    ├── benchmark_merged_*.json       # Merged results for comparison
    ├── comparison_*.json             # Regression comparison results
    └── summary_*.txt                 # Summary reports
```

## Synthetic Data Generation

The `generate_synthetic_data.py` creates a ~5MB ROOT file for testing:

```bash
# Generate with defaults (100k rows)
python generate_synthetic_data.py

# Custom size
python generate_synthetic_data.py --rows 500000 --tracks 50000

# Verify file
python generate_synthetic_data.py --verify
```

**Generated structure:**
- Main tree (`tree`): 100k rows, 10 columns (x, y, z, dy, dz, sec, row, mX, mY, track_idx)
- Subframe (`T`): 10k tracks (track_idx, mP3, mP4, mX, dEdxTPC)

## CI Integration

For continuous integration, use quick mode:

```bash
# In CI script
./run_benchmark.sh --quick

# Or strict mode to fail on regression
./run_benchmark.sh --quick --strict

# Full regression detection workflow
./run_benchmark.sh --quick --compare-baseline --strict
```

### GitHub Actions Example

```yaml
- name: Run benchmarks
  run: |
    cd AliasDataFrame/benchmarks
    ./run_benchmark.sh --quick

- name: Check for regressions
  run: |
    cd AliasDataFrame/benchmarks
    ./run_benchmark.sh --quick --compare-baseline --strict
```

### GitLab CI Example

```yaml
benchmark:
  script:
    - cd AliasDataFrame/benchmarks
    - ./run_benchmark.sh --quick --output artifacts/
  artifacts:
    paths:
      - artifacts/

regression-check:
  script:
    - cd AliasDataFrame/benchmarks
    - ./run_benchmark.sh --quick --compare-baseline --strict
  only:
    - merge_requests
```

## Troubleshooting

### Benchmark is slow

1. **Run diagnostic first:** `python diagnose_read_performance.py data.root`
2. Check server load: `top`, `htop`
3. Try `num_workers=1` to isolate parallel issues
4. Use `--quick` mode for faster iteration

### Timeouts in parallel benchmark

1. Reduce `--max-workers`
2. Increase `--timeout`
3. Check if file is on network filesystem (NFS/AFS)
4. Consider copying data to local disk first

### High variance in results

1. Server is likely under load from other users
2. Run on dedicated node or during off-peak hours
3. Use batch system for reproducible results

### Results differ from baseline

1. Check if code changed (regression)
2. Check if hardware/environment changed
3. Re-establish baseline with `--save-baseline`

### Unexpected 100x slowdown

This is usually caused by external factors, not code issues:

1. **Run diagnostic:** `python diagnose_read_performance.py data.root --json diag.json`
2. Check diagnostic results for:
   - Worker scaling (thread contention?)
   - Cache effect (cold start?)
   - Network vs local (I/O bottleneck?)
3. Common causes:
   - Shared server under heavy load
   - Network filesystem congestion
   - Cold cache after server restart
   - Parallel job contention
