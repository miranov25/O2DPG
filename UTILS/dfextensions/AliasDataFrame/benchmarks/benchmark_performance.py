#!/usr/bin/env python3
"""
benchmark_performance.py - Synthetic performance benchmarks for AliasDataFrame

No external files needed. Tests core operations at scale.

Usage:
    python benchmark_performance.py                    # Full benchmark (1M rows)
    python benchmark_performance.py --quick            # Quick mode (100k rows)
    python benchmark_performance.py --json results.json  # Output to JSON
    python benchmark_performance.py --update-baselines # Save current as baseline

Exit Codes:
    0 - Always (failures are reported, not fatal)
    
Thresholds (1M rows):
    create_adf:      < 0.5s
    add_aliases:     < 1.0s
    validate_schema: < 1.0s
    materialize:     < 0.5s
    compress:        < 2.0s
    export_schema:   < 0.5s
"""

import argparse
import gc
import json
import os
import platform
import sys
import time
import tracemalloc
from datetime import datetime

import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AliasDataFrame import AliasDataFrame


# =============================================================================
# Configuration
# =============================================================================

# Thresholds for 1M rows (seconds)
THRESHOLDS_FULL = {
    'create_adf': 0.5,
    'add_aliases': 1.0,
    'validate_schema': 1.0,
    'materialize': 0.5,
    'compress': 2.0,
    'export_schema': 0.5,
}

# Quick mode configuration
QUICK_ROWS = 100_000
QUICK_ALIASES = 10
FULL_ROWS = 1_000_000
FULL_ALIASES = 50

# Baseline file location
BASELINES_FILE = os.path.join(os.path.dirname(__file__), 'baselines.json')


# =============================================================================
# Utility Functions
# =============================================================================

def calculate_thresholds(n_rows, base_rows=FULL_ROWS, base_thresholds=THRESHOLDS_FULL):
    """
    Calculate thresholds for given row count.
    
    Formula: threshold = base_threshold * (n_rows / base_rows) + 0.05
    The 0.05s offset accounts for fixed overhead.
    """
    scale = n_rows / base_rows
    return {name: max(0.1, threshold * scale + 0.05) 
            for name, threshold in base_thresholds.items()}


def create_synthetic_data(n_rows):
    """Create synthetic DataFrame matching typical TPC calibration schema."""
    np.random.seed(42)
    return pd.DataFrame({
        'x': np.random.randn(n_rows).astype('float32') * 100 + 200,
        'y': np.random.randn(n_rows).astype('float32') * 10,
        'z': np.random.randn(n_rows).astype('float32') * 200,
        'dy': np.random.randn(n_rows).astype('float16'),
        'dz': np.random.randn(n_rows).astype('float16'),
        'sec': np.random.randint(0, 36, n_rows, dtype='uint8'),
        'row': np.random.randint(0, 152, n_rows, dtype='uint8'),
        'mP3': np.random.randn(n_rows).astype('float32'),
        'mP4': np.random.randn(n_rows).astype('float32'),
        'track_idx': np.random.randint(0, max(1, n_rows // 100), n_rows, dtype='int32'),
    })


def measure_memory(func):
    """Decorator to measure peak memory usage."""
    def wrapper(*args, **kwargs):
        gc.collect()
        tracemalloc.start()
        
        result = func(*args, **kwargs)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return result, peak / (1024 * 1024)  # Convert to MB
    return wrapper


def format_bar(value, max_value, width=30):
    """Create a visual bar for progress display."""
    filled = int(width * min(value / max_value, 1.0))
    return '█' * filled + '░' * (width - filled)


def load_baselines():
    """Load baseline results from file."""
    if os.path.exists(BASELINES_FILE):
        with open(BASELINES_FILE, 'r') as f:
            return json.load(f)
    return None


def save_baselines(results):
    """Save current results as baselines."""
    baselines = {
        'timestamp': datetime.now().isoformat(),
        'results': {name: data['time_s'] for name, data in results.items()}
    }
    with open(BASELINES_FILE, 'w') as f:
        json.dump(baselines, f, indent=2)
    print(f"Baselines saved to {BASELINES_FILE}")


# =============================================================================
# Benchmark Functions
# =============================================================================

def benchmark_create_adf(df):
    """Benchmark AliasDataFrame creation."""
    gc.collect()
    tracemalloc.start()
    
    t0 = time.perf_counter()
    adf = AliasDataFrame(df)
    elapsed = time.perf_counter() - t0
    
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return adf, elapsed, peak / (1024 * 1024)


def benchmark_add_aliases(adf, n_aliases):
    """Benchmark adding multiple aliases."""
    gc.collect()
    tracemalloc.start()
    
    t0 = time.perf_counter()
    for i in range(n_aliases):
        adf.add_alias(f'alias_{i}', f'x + y * {i} + z', dtype=np.float32)
    elapsed = time.perf_counter() - t0
    
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return elapsed, peak / (1024 * 1024)


def benchmark_validate_schema(adf):
    """Benchmark schema validation."""
    gc.collect()
    tracemalloc.start()
    
    t0 = time.perf_counter()
    adf.validate_schema(verbose=False)
    elapsed = time.perf_counter() - t0
    
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return elapsed, peak / (1024 * 1024)


def benchmark_materialize(adf, alias_name='alias_0'):
    """Benchmark alias materialization."""
    gc.collect()
    tracemalloc.start()
    
    t0 = time.perf_counter()
    adf.materialize_alias(alias_name)
    elapsed = time.perf_counter() - t0
    
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return elapsed, peak / (1024 * 1024)


def benchmark_compress(adf):
    """Benchmark column compression."""
    spec = {
        'dy': {
            'compress': 'round(asinh(dy)*40)',
            'decompress': 'sinh(dy_c/40.)',
            'compressed_col': 'dy_c',
            'compressed_dtype': np.int16,
            'decompressed_dtype': np.float16
        }
    }
    
    gc.collect()
    tracemalloc.start()
    
    t0 = time.perf_counter()
    adf.compress_columns(spec)
    elapsed = time.perf_counter() - t0
    
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return elapsed, peak / (1024 * 1024)


def benchmark_export_schema(adf):
    """Benchmark schema export."""
    gc.collect()
    tracemalloc.start()
    
    t0 = time.perf_counter()
    schema = adf.export_definition_schema()
    elapsed = time.perf_counter() - t0
    
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return elapsed, peak / (1024 * 1024)


# =============================================================================
# Main Benchmark Runner
# =============================================================================

def run_benchmarks(n_rows=FULL_ROWS, n_aliases=FULL_ALIASES, verbose=True):
    """
    Run all synthetic benchmarks.
    
    Parameters
    ----------
    n_rows : int
        Number of rows in synthetic DataFrame
    n_aliases : int
        Number of aliases to create
    verbose : bool
        Print progress to stdout
        
    Returns
    -------
    dict : Results with timing and memory for each operation
    """
    thresholds = calculate_thresholds(n_rows)
    results = {}
    
    if verbose:
        print("=" * 70)
        print("SYNTHETIC PERFORMANCE BENCHMARK")
        print("=" * 70)
        print(f"Rows:      {n_rows:,}")
        print(f"Aliases:   {n_aliases}")
        print(f"Hostname:  {platform.node()}")
        print(f"Python:    {platform.python_version()}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        print()
    
    # 1. Create synthetic data
    if verbose:
        print("--- 1. Create synthetic data ---")
    t0 = time.perf_counter()
    df = create_synthetic_data(n_rows)
    df_time = time.perf_counter() - t0
    if verbose:
        print(f"  DataFrame created: {df_time:.3f}s")
        print(f"  Shape: {df.shape}")
        print(f"  Memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
    
    # 2. ADF creation
    if verbose:
        print("\n--- 2. AliasDataFrame creation ---")
    adf, elapsed, memory = benchmark_create_adf(df)
    results['create_adf'] = {
        'time_s': elapsed,
        'memory_mb': memory,
        'threshold_s': thresholds['create_adf'],
        'passed': elapsed < thresholds['create_adf']
    }
    if verbose:
        status = '✓' if results['create_adf']['passed'] else '✗'
        print(f"  {status} Time: {elapsed:.3f}s (threshold: {thresholds['create_adf']:.2f}s)")
        print(f"    Memory: {memory:.1f} MB")
    
    # 3. Add aliases
    if verbose:
        print(f"\n--- 3. Add {n_aliases} aliases ---")
    elapsed, memory = benchmark_add_aliases(adf, n_aliases)
    results['add_aliases'] = {
        'time_s': elapsed,
        'memory_mb': memory,
        'threshold_s': thresholds['add_aliases'],
        'passed': elapsed < thresholds['add_aliases'],
        'count': n_aliases,
        'per_alias_ms': elapsed / n_aliases * 1000
    }
    if verbose:
        status = '✓' if results['add_aliases']['passed'] else '✗'
        print(f"  {status} Time: {elapsed:.3f}s (threshold: {thresholds['add_aliases']:.2f}s)")
        print(f"    Per alias: {results['add_aliases']['per_alias_ms']:.2f}ms")
        print(f"    Memory: {memory:.1f} MB")
    
    # 4. Validate schema
    if verbose:
        print("\n--- 4. Validate schema ---")
    elapsed, memory = benchmark_validate_schema(adf)
    results['validate_schema'] = {
        'time_s': elapsed,
        'memory_mb': memory,
        'threshold_s': thresholds['validate_schema'],
        'passed': elapsed < thresholds['validate_schema']
    }
    if verbose:
        status = '✓' if results['validate_schema']['passed'] else '✗'
        print(f"  {status} Time: {elapsed:.3f}s (threshold: {thresholds['validate_schema']:.2f}s)")
        print(f"    Memory: {memory:.1f} MB")
    
    # 5. Materialize
    if verbose:
        print("\n--- 5. Materialize alias ---")
    elapsed, memory = benchmark_materialize(adf)
    results['materialize'] = {
        'time_s': elapsed,
        'memory_mb': memory,
        'threshold_s': thresholds['materialize'],
        'passed': elapsed < thresholds['materialize']
    }
    if verbose:
        status = '✓' if results['materialize']['passed'] else '✗'
        print(f"  {status} Time: {elapsed:.3f}s (threshold: {thresholds['materialize']:.2f}s)")
        print(f"    Memory: {memory:.1f} MB")
    
    # 6. Compression
    if verbose:
        print("\n--- 6. Compress column ---")
    elapsed, memory = benchmark_compress(adf)
    results['compress'] = {
        'time_s': elapsed,
        'memory_mb': memory,
        'threshold_s': thresholds['compress'],
        'passed': elapsed < thresholds['compress']
    }
    if verbose:
        status = '✓' if results['compress']['passed'] else '✗'
        print(f"  {status} Time: {elapsed:.3f}s (threshold: {thresholds['compress']:.2f}s)")
        print(f"    Memory: {memory:.1f} MB")
    
    # 7. Export schema
    if verbose:
        print("\n--- 7. Export schema ---")
    elapsed, memory = benchmark_export_schema(adf)
    results['export_schema'] = {
        'time_s': elapsed,
        'memory_mb': memory,
        'threshold_s': thresholds['export_schema'],
        'passed': elapsed < thresholds['export_schema']
    }
    if verbose:
        status = '✓' if results['export_schema']['passed'] else '✗'
        print(f"  {status} Time: {elapsed:.3f}s (threshold: {thresholds['export_schema']:.2f}s)")
        print(f"    Memory: {memory:.1f} MB")
    
    # Cleanup
    del adf, df
    gc.collect()
    
    return results


def print_summary(results, baselines=None):
    """Print summary of benchmark results."""
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    total_time = sum(r['time_s'] for r in results.values())
    total_memory = sum(r['memory_mb'] for r in results.values())
    all_passed = all(r['passed'] for r in results.values())
    
    # Header
    if baselines:
        print(f"{'Operation':<20} {'Time (s)':<12} {'Threshold':<12} {'Baseline':<12} {'Status':<8}")
    else:
        print(f"{'Operation':<20} {'Time (s)':<12} {'Threshold':<12} {'Memory (MB)':<12} {'Status':<8}")
    print("-" * 70)
    
    # Results
    for name, data in results.items():
        status = '✓ PASS' if data['passed'] else '✗ FAIL'
        time_str = f"{data['time_s']:.3f}"
        threshold_str = f"{data['threshold_s']:.2f}"
        
        if baselines and name in baselines.get('results', {}):
            baseline_str = f"{baselines['results'][name]:.3f}"
            ratio = data['time_s'] / baselines['results'][name]
            if ratio > 2.0:
                status = f'⚠ {ratio:.1f}x'
            print(f"{name:<20} {time_str:<12} {threshold_str:<12} {baseline_str:<12} {status:<8}")
        else:
            memory_str = f"{data['memory_mb']:.1f}"
            print(f"{name:<20} {time_str:<12} {threshold_str:<12} {memory_str:<12} {status:<8}")
    
    print("-" * 70)
    print(f"{'TOTAL':<20} {total_time:<12.3f} {'':<12} {total_memory:<12.1f}")
    
    # Final status
    print("\n" + "=" * 70)
    if all_passed:
        print("✓ ALL BENCHMARKS PASSED")
    else:
        failed = [name for name, data in results.items() if not data['passed']]
        print(f"✗ FAILED: {', '.join(failed)}")
    print("=" * 70)
    
    return all_passed


def export_json(results, filepath, n_rows, mode):
    """Export results to JSON file."""
    output = {
        'timestamp': datetime.now().isoformat(),
        'hostname': platform.node(),
        'python_version': platform.python_version(),
        'platform': platform.platform(),
        'rows': n_rows,
        'mode': mode,
        'results': results,
        'all_passed': all(r['passed'] for r in results.values()),
        'total_time_s': sum(r['time_s'] for r in results.values()),
        'total_memory_mb': sum(r['memory_mb'] for r in results.values()),
    }
    
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults exported to: {filepath}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Synthetic performance benchmarks for AliasDataFrame",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python benchmark_performance.py                     # Full benchmark
    python benchmark_performance.py --quick             # Quick mode (CI)
    python benchmark_performance.py --json out.json    # Export to JSON
    python benchmark_performance.py --update-baselines # Save as baseline
        """
    )
    parser.add_argument('--quick', action='store_true',
                        help=f'Quick mode: {QUICK_ROWS:,} rows, {QUICK_ALIASES} aliases')
    parser.add_argument('--rows', type=int, default=None,
                        help='Custom row count (overrides --quick)')
    parser.add_argument('--json', type=str, metavar='FILE',
                        help='Export results to JSON file')
    parser.add_argument('--update-baselines', action='store_true',
                        help='Save current results as baseline')
    parser.add_argument('--quiet', action='store_true',
                        help='Minimal output')
    
    args = parser.parse_args()
    
    # Determine configuration
    if args.rows:
        n_rows = args.rows
        n_aliases = max(10, args.rows // 20000)  # Scale aliases with rows
        mode = 'custom'
    elif args.quick:
        n_rows = QUICK_ROWS
        n_aliases = QUICK_ALIASES
        mode = 'quick'
    else:
        n_rows = FULL_ROWS
        n_aliases = FULL_ALIASES
        mode = 'full'
    
    # Run benchmarks
    results = run_benchmarks(n_rows=n_rows, n_aliases=n_aliases, verbose=not args.quiet)
    
    # Load baselines for comparison
    baselines = load_baselines()
    
    # Print summary
    if not args.quiet:
        all_passed = print_summary(results, baselines)
    else:
        all_passed = all(r['passed'] for r in results.values())
        print(f"{'PASS' if all_passed else 'FAIL'}: {sum(r['time_s'] for r in results.values()):.2f}s")
    
    # Export to JSON if requested
    if args.json:
        export_json(results, args.json, n_rows, mode)
    
    # Update baselines if requested
    if args.update_baselines:
        save_baselines(results)
    
    # Always return 0 (report failures, don't fail CI)
    return 0


if __name__ == '__main__':
    sys.exit(main())
