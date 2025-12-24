# Benchmark Framework v1.0

Standardized benchmarking for `dfextensions` subprojects with automated regression detection.

## Overview

The benchmark framework provides:

- **Reproducible benchmarks** with configurable warmup and run counts
- **Memory profiling** using RSS (Resident Set Size) tracking
- **Environment fingerprinting** including Python, NumPy, and Numba versions
- **Historical tracking** with JSON storage and DataFrame loading
- **Regression detection** using median baselines with configurable thresholds
- **Trend visualization** and HTML/text report generation

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

## CLI Reference

```
python -m dfextensions.benchmarks.runner [OPTIONS]

Required:
  --subproject NAME     Subproject to benchmark (e.g., groupby_regression)

Optional:
  --suite SUITE         Benchmark suite: quick (default), release
  --n-runs N            Number of timed runs per benchmark (default: 3)
  --warmup-runs N       Warmup runs before timing (default: 2)
  --profile             Enable tracemalloc memory profiling
  --check-only          Check regressions without running new benchmarks
  --dry-run             Run but don't save results
  --quiet               Minimal output
  --time-threshold F    Time regression threshold (default: 0.10 = 10%)
  --memory-threshold F  Memory regression threshold (default: 0.15 = 15%)
  --cross-env           Allow cross-environment baseline comparisons
```

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | All benchmarks passed, no regressions |
| 1 | Regression detected |
| 2 | Execution error |

## Storage Layout

```
$BENCHMARK_PREFIX/                        # Default: ~/benchmark_results
├── 2024-12-23T14-30-00/                 # Timestamp directory
│   └── groupby_regression/              # Subproject
│       ├── results.json                 # Benchmark results
│       └── alarms.json                  # Regression alarms (if any)
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

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2024-12-23 | Initial release: Core + Analysis + Polish |

## License

Part of the ALICE O2 data processing framework.
