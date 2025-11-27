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

## Overview

| Script | Purpose | Data Required |
|--------|---------|---------------|
| `run_benchmark.sh` | **Main entry point** - runs all benchmarks | Auto-generates |
| `benchmark_performance.py` | Core operations timing | Synthetic (built-in) |
| `benchmark_read_tree.py` | ROOT file read tests | ROOT file |
| `benchmark_subframe.py` | Subframe join tests | ROOT file |
| `benchmark_parallel.py` | Worker scaling analysis | ROOT file |
| `generate_synthetic_data.py` | Create test ROOT file | None |

## run_benchmark.sh (Main Entry Point)

Runs ALL benchmarks with pytest-style output:

```
==================== BENCHMARK SESSION STARTS ====================
Timestamp:  2025-11-27T17:30:00
Host:       your-machine
Python:     Python 3.9.6

✓ benchmark_performance.py          PASSED (0.06s)
✓ benchmark_read_tree.py            PASSED (2.34s)
✓ benchmark_subframe.py             PASSED (1.20s)
✓ benchmark_parallel.py             PASSED (8.50s)

==================== 4 passed, 0 failed in 12.10s ====================
```

### Options

```bash
./run_benchmark.sh                 # Full benchmarks
./run_benchmark.sh --quick         # Quick mode (smaller data)
./run_benchmark.sh --synthetic-only # Only synthetic (no ROOT)
./run_benchmark.sh --generate-data  # Force regenerate test data
./run_benchmark.sh --strict        # Exit 1 on failure
./run_benchmark.sh --verbose       # Show detailed output
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
8-branch-conv   2.1          180          95
4-aliasdf       2.3          190          95

======================================================================
RECOMMENDATIONS
======================================================================
Fastest approach: 8-branch-thr-conv (2.1s)
✓✓ THREADED BRANCH-BY-BRANCH is the winner!
   → Use this as default with num_workers>1
======================================================================
```

### Key Insights

This benchmark helped identify that **threaded branch-by-branch reading** is optimal:
- 4-6x faster than one-shot reading
- 50% less peak memory
- Now the default in `AliasDataFrame.read_tree()`

## benchmark_subframe.py

Validates subframe functionality: loading, joining, and correctness.

### Purpose

- Verify subframe detection and loading
- Validate join correctness (invariant tests)
- Measure alias materialization speed
- Check missing key statistics

### Tests Performed

| Test | Description |
|------|-------------|
| File Loading | Load ROOT file with `read_tree()` |
| Subframe Detection | Detect and load all subframes |
| Join Correctness | Verify `T.column` lookups return correct values |
| Invariant Test | Same key must give same value (std within group = 0) |
| Materialization Speed | Time to materialize subframe aliases |
| Missing Key Stats | Coverage analysis (main keys vs subframe keys) |

### Usage

```bash
# Basic run
python benchmark_subframe.py data.root

# Custom tree name
python benchmark_subframe.py data.root --treename mytree

# Limit entries
python benchmark_subframe.py data.root --entries 100000

# Custom workers
python benchmark_subframe.py data.root --workers 4
```

### Output

```
============================================================
AliasDataFrame Subframe Benchmark
============================================================
File: data.root
Tree: tree
Max entries: 100000

--- 1. File Loading ---
  ✓ Loaded: 100,000 rows, 45 columns
  ✓ Memory: 38.5 MB
  ✓ Time: 1.23 s
  ✓ Speed: 81,300 rows/sec

--- 2. Subframe Detection ---
  Found 1 subframes: ['T']
  ✓ T: 5,234 rows, index=['track_idx']
    Columns: ['track_idx', 'mP3', 'mP4', 'mX', 'dEdxTPC']

--- 3. Join Correctness Validation ---
  T.mP3: 98,234/100,000 valid (98.2%)
  T.mX: 98,234/100,000 valid (98.2%)

  Invariant test (same key → same value):
    ✓ T.mP3: max_std_within_group=0.00e+00 (CORRECT)
    ✓ T.mX: max_std_within_group=0.00e+00 (CORRECT)

--- 4. Alias Materialization Speed ---
  ✓ simple_lookup: 45.2 ms, valid=98,234/100,000
  ✓ expression: 52.1 ms, valid=98,234/100,000

--- 5. Missing Key Statistics ---
  T:
    Main keys: 5,500
    Subframe keys: 5,234
    Coverage: 95.2% (5,234 matched)
    Missing in subframe: 266

============================================================
SUMMARY: 8 passed, 0 failed
============================================================
```

### Key Tests Explained

**Invariant Test:** Groups main frame by index key, checks that all rows with the same key get the same subframe value. Standard deviation within each group should be 0 (or near-zero for floating point).

**Coverage:** Percentage of main frame keys that exist in the subframe. <100% is normal (not all clusters belong to tracks).

## JSON Output Format

Both scripts support `--json` output for programmatic processing:

```json
{
  "timestamp": "2025-11-27T16:30:00.123456",
  "hostname": "lxbk1130",
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
├── README.md                      # This file
├── run_benchmark.sh               # Main entry point (pytest-style)
├── generate_synthetic_data.py     # Creates test ROOT file (~5MB)
├── benchmark_performance.py       # Synthetic benchmarks (NEW)
├── benchmark_parallel.py          # Parallel scaling tests (NEW)
├── benchmark_read_tree.py         # ROOT file read comparison (existing)
├── benchmark_subframe.py          # Subframe validation (existing)
├── baselines.json                 # Saved baselines (auto-generated)
├── synthetic_data.root            # Test data (auto-generated, gitignored)
└── results/                       # Output directory (gitignored)
    ├── benchmark_*.json           # Detailed results
    └── summary_*.txt              # Summary reports
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
```

### GitHub Actions Example

```yaml
- name: Run benchmarks
  run: |
    cd AliasDataFrame/benchmarks
    ./run_benchmark.sh --quick
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
```

## Troubleshooting

### Benchmark is slow

1. Check server load: `top`, `htop`
2. Try `num_workers=1` to isolate parallel issues
3. Use `--quick` mode for faster iteration

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
3. Re-establish baseline with `--update-baselines`
