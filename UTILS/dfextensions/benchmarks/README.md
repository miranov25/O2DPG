# Benchmark Framework v1.0

Standardized benchmarking for `dfextensions` subprojects with automated regression detection.

## Overview

The benchmark framework provides:

- **Reproducible benchmarks** with configurable warmup and run counts
- **Memory profiling** using RSS (Resident Set Size) tracking
- **CPU profiling** with cProfile integration and per-benchmark `.prof` files
- **Environment fingerprinting** including Python, NumPy, and Numba versions
- **Historical tracking** with JSON storage and DataFrame loading
- **Regression detection** using median baselines with configurable thresholds
- **Noise analysis** with CV% statistics for alarm threshold tuning
- **Trend visualization** with PNG plots and CLI commands

## Quick Start

### Running Benchmarks

```bash
# Run quick benchmark suite
python -m dfextensions.benchmarks.runner --subproject groupby_regression --suite quick

# Run with profiling (tracemalloc)
python -m dfextensions.benchmarks.runner --subproject groupby_regression --profile

# Check for regressions without running new benchmarks
python -m dfextensions.benchmarks.runner --subproject groupby_regression --check-only
```

### Analyzing History (Phase 12.14c.GB)

```bash
# View benchmark history summary
python -m dfextensions.benchmarks.runner --subproject groupby_regression --history

# View noise statistics (CV%) for alarm tuning
python -m dfextensions.benchmarks.runner --subproject groupby_regression --history-stats

# Generate trend plots
python -m dfextensions.benchmarks.runner --subproject groupby_regression --plot ./plots/
```

### Programmatic Usage

```python
from dfextensions.benchmarks import (
    run_benchmarks,
    load_history,
    detect_regressions,
)

# Run benchmarks
run = run_benchmarks(subproject="groupby_regression", suite="quick")
run.save("results.json")

# Load history and detect regressions
history_df = load_history("groupby_regression")
alarms, results = detect_regressions(run, history_df)

if alarms:
    print(f"⚠ {len(alarms)} regression(s) detected!")
```

### Using AliasDataFrame for Analysis (Phase 12.14c.GB)

```python
from dfextensions.benchmarks.benchmark_adf import (
    load_benchmark_adf,
    compute_benchmark_statistics,
)

# Load history as AliasDataFrame with subframes
adf = load_benchmark_adf("groupby_regression", max_runs=20)

# Main frame: benchmark results
print(adf.df.head())

# Subframes: CPU profiles and memory stats
print(adf.subframes['TopCPU'].df.head())
print(adf.subframes['TopMemory'].df.head())

# Compute noise statistics
stats = compute_benchmark_statistics("groupby_regression", baseline="7d")
print(stats[['benchmark_id', 'mean_time_s', 'cv_pct', 'high_noise']])
```

## CLI Reference

```
python -m dfextensions.benchmarks.runner [OPTIONS]

Required:
  --subproject NAME     Subproject to benchmark (e.g., groupby_regression)

Benchmark Execution:
  --suite SUITE         Benchmark suite: quick (default), release
  --n-runs N            Number of timed runs per benchmark (default: 3)
  --warmup-runs N       Warmup runs before timing (default: 2)
  --profile             Enable tracemalloc memory profiling
  --no-profile          Disable cProfile capture (faster for CI)
  --check-only          Check regressions without running new benchmarks
  --dry-run             Run but don't save results
  --quiet               Minimal output

Thresholds:
  --time-threshold F    Time regression threshold (default: 0.10 = 10%)
  --memory-threshold F  Memory regression threshold (default: 0.15 = 15%)
  --cross-env           Allow cross-environment baseline comparisons

Visualization (Phase 12.14c.GB):
  --history             Show benchmark history summary
  --history-stats       Show noise statistics (mean, std, CV%) per benchmark
  --plot DIR            Generate trend plots to specified directory
  --baseline RANGE      Baseline range for statistics (default: 7d)
  --max-runs N          Maximum runs to load (default: all)
```

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | All benchmarks passed, no regressions / Visualization success |
| 1 | Regression detected / No data found |
| 2 | Execution error / Partial success (some plots failed) |
| 3 | Dependency missing (matplotlib/pyyaml) |

## Noise Analysis

### Understanding CV% (Coefficient of Variation)

The `--history-stats` command shows noise statistics for each benchmark:

```
Benchmark Statistics: groupby_regression (baseline: 7d)
════════════════════════════════════════════════════════════════════
  Benchmark                               Mean      Std     CV%    N
  ──────────────────────────────────────────────────────────────────
  kernel_multi_fit:K1                    3.6ms    4.1ms  114.4%    5  ⚠ HIGH
  v5_batch_fit:S2:n_jobs=4              44.7ms    2.1ms    4.6%    5
  ──────────────────────────────────────────────────────────────────

  Summary: 15/18 benchmarks have high noise (CV > 10%)
```

### Interpreting Results

| CV% | Classification | Recommendation |
|-----|----------------|----------------|
| < 5% | Low noise | Standard thresholds work well |
| 5-10% | Moderate | Monitor for trends |
| > 10% | High noise | Consider median-based thresholds |
| > 50% | Very high | Micro-benchmark, expect variance |

### Why High CV?

Sub-millisecond benchmarks inherently have high CV because:
- Timer resolution limits (~1ms on some systems)
- CPU frequency scaling
- Cache effects
- Context switching

**Recommendation:** For micro-benchmarks with CV > 50%, consider:
1. Using median instead of mean for baselines
2. Increasing sample count (`--n-runs 10`)
3. Running on dedicated hardware

## Trend Plots

### Generated Plots

The `--plot DIR` command generates these PNG files:

| Plot | Description | Grouping |
|------|-------------|----------|
| `time_trend.png` | Execution time over time | By benchmark name |
| `wall_time_trend.png` | Wall time (includes setup) | By benchmark name |
| `rss_trend.png` | Peak RSS memory | By benchmark name |
| `throughput_trend.png` | Rows/sec throughput | By benchmark ID |
| `time_distribution.png` | Histogram of execution times | All benchmarks |

### Customizing Plots

Plot specifications are defined in `benchmarks/specs/benchmark_specs.yaml`:

```yaml
specs:
  - name: time_trend
    title: "Execution Time Trend"
    x: timestamp
    y: time_s
    kind: scatter
    groupby: name
    enabled: true
```

Disable a plot by setting `enabled: false`.

## Storage Layout

```
$BENCHMARK_PREFIX/                        # Default: ~/benchmark_results
├── 2024-12-23T14-30-00/                 # Timestamp directory
│   └── groupby_regression/              # Subproject
│       ├── results.json                 # Benchmark results
│       ├── alarms.json                  # Regression alarms (if any)
│       └── profiles/                    # cProfile data (Phase 12.14b)
│           ├── v5_batch_fit_S1_n_jobs_1.prof
│           └── v5_batch_fit_S1_n_jobs_4.prof
└── 2024-12-24T10-15-30/
    └── groupby_regression/
        └── results.json
```

Set custom storage location:

```bash
export BENCHMARK_PREFIX=/path/to/benchmarks
```

## JSON Schema

### results.json

```json
{
  "schema_version": 1,
  "meta": {
    "run_id": "20241223-143000-123_abc123_456",
    "timestamp": "2024-12-23T14:30:00.123Z",
    "commit": "abc123",
    "branch": "main",
    "env_id": "Darwin_M2_3.9.6_np1.24.0_nb0.58.0",
    "subproject": "groupby_regression",
    "run_mode": "gate",
    "suite": "quick",
    "hostname": "macbook.local"
  },
  "summary": {
    "n_benchmarks": 8,
    "n_passed": 8,
    "n_failed": 0,
    "n_regressions": 0,
    "total_time_s": 45.2
  },
  "benchmarks": [
    {
      "id": "v5_batch_fit:S2:n_jobs=4",
      "name": "v5_batch_fit",
      "scenario": "S2",
      "params": {"n_jobs": 4, "n_rows": 100000, "n_fits": 1000},
      "time_s": 0.312,
      "time_std_s": 0.015,
      "wall_time_s": 0.350,
      "n_runs": 3,
      "peak_rss_mb": 605.2,
      "throughput_rows_per_sec": 320512.8,
      "status": "OK"
    }
  ],
  "alarms": []
}
```

### Environment ID Format

```
{platform}_{cpu}_{python}_{numpy}_{numba}

Examples:
  Darwin_M2_3.9.6_np1.24.0_nb0.58.0
  Linux_x86_64_3.11.0_np1.26.0_nb0.59.0
```

## Regression Detection

### Algorithm

1. Load history for same `env_id` (prevents cross-platform false alarms)
2. Exclude `run_mode="profile"` runs (different overhead characteristics)
3. Calculate baseline using **median** of last 20 runs (robust to outliers)
4. Require minimum 3 samples before triggering alarms (`MIN_BASELINE_SAMPLES`)
5. Compare current result against baseline
6. Trigger alarm if threshold exceeded

### Thresholds

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| Time | 10% | Standard for performance regression |
| Memory | 15% | Higher due to RSS variance |

### Minimum History

Regression detection requires at least 3 historical samples. This prevents false positives during the first few runs of a new benchmark.

## Memory Measurement (RSS)

### What is RSS?

RSS (Resident Set Size) is the portion of process memory held in RAM. The framework measures **peak RSS** during benchmark execution.

### Important Semantics

1. **Process-wide measurement**: RSS reflects the entire process, not isolated per-benchmark
2. **Non-decreasing within a run**: Memory freed by Python GC may not reduce RSS immediately
3. **Cross-run comparisons valid**: Same benchmark across different runs is comparable
4. **Within-run ordering effects**: Benchmark execution order may affect absolute values

### Why 15% Threshold for Memory?

RSS measurements have higher variance than timing due to:
- Python garbage collection timing
- Shared memory pages
- Memory allocator behavior
- Background process activity

A 15% threshold balances sensitivity against noise.

## CPU Profiling (Phase 12.14b)

### Automatic cProfile Capture

By default, each benchmark captures a cProfile and saves it as a `.prof` file:

```
profiles/
├── v5_batch_fit_S1_n_jobs_1.prof
├── v5_batch_fit_S1_n_jobs_4.prof
└── kernel_single_fit_K1.prof
```

### Analyzing Profiles

```python
import pstats

# Load and display top functions
stats = pstats.Stats("v5_batch_fit_S1_n_jobs_4.prof")
stats.sort_stats("cumulative")
stats.print_stats(20)
```

### Disabling cProfile

For faster CI runs:

```bash
python -m dfextensions.benchmarks.runner --subproject groupby_regression --no-profile
```

## Platform Support

| Platform | Status | Notes |
|----------|--------|-------|
| macOS (Darwin) | ✅ Full | Tested on M1/M2 |
| Linux | ✅ Full | Tested on Ubuntu 24 |
| Windows | ⚠️ Limited | No RSS (resource module unavailable) |

### Python Version

- **Minimum**: Python 3.9
- **Tested**: 3.9.6 (macOS), 3.12.3 (Linux)

## Visualization

### Trend Plots

```python
from dfextensions.benchmarks import load_history
from dfextensions.benchmarks.report import plot_benchmark_trends

df = load_history("groupby_regression")
fig = plot_benchmark_trends(df, benchmark_id="v5_batch_fit:S2:n_jobs=4")
fig.savefig("trend.png")
```

### HTML Reports

```python
from dfextensions.benchmarks.report import generate_report, save_report

report = generate_report(df, subproject="groupby_regression", current_run=run)
save_report(report, "report.html")
```

## Adding Benchmarks

### 1. Create Scenarios

```python
# dfextensions/groupby_regression/benchmarks/scenarios.py

SCENARIOS = {
    "S1": {"n_rows": 50_000, "n_fits": 1000},
    "S2": {"n_rows": 100_000, "n_fits": 1000},
    "S3": {"n_rows": 200_000, "n_fits": 1000},
}

def create_test_data(scenario: str, seed: int = 42) -> pd.DataFrame:
    params = SCENARIOS[scenario]
    # Generate test data...
    return df
```

### 2. Define Benchmarks

```python
# dfextensions/groupby_regression/benchmarks/bench_v5.py

def get_benchmarks(suite: str = "quick") -> List[dict]:
    return [
        {
            "name": "v5_batch_fit",
            "func": run_v5_batch_fit,
            "scenarios": ["S1", "S2"] if suite == "quick" else ["S1", "S2", "S3"],
            "n_jobs_list": [1, 4],
        },
    ]

def run_v5_batch_fit(scenario: str, n_jobs: int) -> dict:
    df = create_test_data(scenario)
    result = groupby_regression_v5(df, n_jobs=n_jobs)
    return {"n_rows": len(df), "n_fits": len(result)}
```

### 3. Register Subproject

Update `runner.py`:

```python
def discover_benchmarks(subproject: str, suite: str = "quick"):
    if subproject == "groupby_regression":
        from groupby_regression.benchmarks.bench_v5 import get_benchmarks
        return get_benchmarks(suite=suite)
    elif subproject == "alias_dataframe":
        from alias_dataframe.benchmarks.bench_adf import get_benchmarks
        return get_benchmarks(suite=suite)
```

## API Reference

### Core Classes

```python
from dfextensions.benchmarks import (
    # Schema
    BenchmarkResult,
    BenchmarkRun,
    RunMeta,
    Alarm,
    
    # Profiler
    get_peak_rss_mb,
    run_benchmark_with_memory,
    
    # Runner
    run_benchmarks,
    run_single_benchmark,
    
    # History
    load_history,
    filter_by_env,
    get_baseline,
    
    # Regression
    detect_regressions,
    MIN_BASELINE_SAMPLES,
)

# Phase 12.14c.GB: AliasDataFrame integration
from dfextensions.benchmarks.benchmark_adf import (
    load_benchmark_adf,
    compute_benchmark_statistics,
)

# Phase 12.14c.GB: Visualization specs
from dfextensions.benchmarks.specs import (
    load_benchmark_specs,
    get_enabled_specs,
)
```

### Constants

```python
SCHEMA_VERSION = 1
RUNNER_VERSION = "1.0.0"
DEFAULT_WARMUP_RUNS = 2
DEFAULT_N_RUNS = 3
DEFAULT_TIME_THRESHOLD = 0.10  # 10%
DEFAULT_MEMORY_THRESHOLD = 0.15  # 15%
MIN_BASELINE_SAMPLES = 3
```

## CI/CD Integration

### GitHub Actions Example

```yaml
- name: Run Benchmarks
  run: |
    python -m dfextensions.benchmarks.runner \
      --subproject groupby_regression \
      --suite quick
  
- name: Check Exit Code
  run: |
    # Exit code 1 = regression, fail the build
    if [ $? -eq 1 ]; then
      echo "Performance regression detected!"
      exit 1
    fi

- name: Generate Plots (optional)
  run: |
    python -m dfextensions.benchmarks.runner \
      --subproject groupby_regression \
      --plot ./benchmark_plots/
  
- name: Upload Plots
  uses: actions/upload-artifact@v3
  with:
    name: benchmark-plots
    path: ./benchmark_plots/
```

### Jenkins Example

```groovy
stage('Benchmark') {
    steps {
        sh '''
            python -m dfextensions.benchmarks.runner \
                --subproject groupby_regression
        '''
    }
    post {
        failure {
            echo 'Performance regression detected!'
        }
    }
}
```

## Troubleshooting

### "No history found for regression detection"

First run for this subproject/environment. Run more benchmarks to build history.

### "ModuleNotFoundError: No module named 'resource'"

Windows platform - RSS measurement not available. Timing still works.

### "PyYAML required for benchmark specs"

Install PyYAML for `--plot` command:

```bash
pip install pyyaml
```

### "matplotlib required for --plot"

Install matplotlib for plot generation:

```bash
pip install matplotlib
```

### False Positives on Shared Servers

Use `env_id` filtering (default) to prevent cross-machine comparisons. If running on heterogeneous batch farm, consider:

1. Tagging runs by machine class
2. Using dedicated benchmark nodes
3. Increasing thresholds for noisy environments

### Memory Values Seem High

RSS is process-wide and non-decreasing. For accurate per-benchmark memory:
- Use `--profile` mode (tracemalloc)
- Run benchmarks in separate processes
- Consider the memory threshold (15%)

### High CV% Values

Use `--history-stats` to identify noisy benchmarks. For benchmarks with CV > 50%:
- Increase `--n-runs` for more samples
- Consider median-based thresholds
- Accept higher variance for micro-benchmarks

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2024-12-23 | Initial release: Core + Analysis + Polish |
| 1.1.0 | 2026-01-03 | Phase 12.14b: cProfile integration, dual timing |
| 1.2.0 | 2026-01-04 | Phase 12.14c.GB: AliasDataFrame, visualization CLI |

## License

Part of the ALICE O2 data processing framework.
