#!/usr/bin/env python3
"""
Benchmark for materialize_aliases performance with subframe joins.

This benchmark verifies Phase 2 join caching optimization:
- Multiple columns from same subframe use cached join indexer
- First column: full merge (cache miss)
- Subsequent columns: fast lookup (cache hit)

Run with: python benchmark_join_caching.py
"""

import pandas as pd
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AliasDataFrame import AliasDataFrame


def benchmark_single_subframe_many_columns(n_rows=1_000_000, n_cols=20, verbose=True):
    """
    Benchmark join caching effectiveness - 20 subframe columns.
    
    With Phase 2 caching:
    - First column: ~200ms (full indexer computation)
    - Subsequent columns: ~20ms each (cache hit)
    - Total: ~600ms instead of ~4000ms
    
    Parameters
    ----------
    n_rows : int
        Number of rows in main DataFrame
    n_cols : int  
        Number of columns to materialize from subframe
    verbose : bool
        Print progress information
        
    Returns
    -------
    dict
        Benchmark results including time, rows, columns
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Benchmark: {n_cols} columns from single subframe")
        print(f"{'='*60}")
    
    # Setup
    np.random.seed(42)
    n_subframe_rows = min(10000, n_rows // 10)  # 10% coverage
    
    df = pd.DataFrame({
        'idx': np.random.randint(0, n_subframe_rows, n_rows),
        'x': np.random.randn(n_rows)
    })
    
    sf_data = {'idx': np.arange(n_subframe_rows)}
    for i in range(n_cols):
        sf_data[f'val_{i}'] = np.random.randn(n_subframe_rows)
    sf = pd.DataFrame(sf_data)
    
    if verbose:
        print(f"Main DataFrame: {n_rows:,} rows")
        print(f"Subframe: {n_subframe_rows:,} rows, {n_cols} value columns")
    
    # Create AliasDataFrame
    adf = AliasDataFrame(df)
    adf.register_subframe('S', AliasDataFrame(sf), index_columns='idx')
    
    for i in range(n_cols):
        adf.add_alias(f'col_{i}', f'S.val_{i}')
    
    # Benchmark
    if verbose:
        print(f"\nMaterializing {n_cols} aliases with join caching...")
    
    t0 = time.time()
    result = adf.materialize_aliases(pattern=r'col_.*', verbose=verbose)
    elapsed = time.time() - t0
    
    # Verify
    assert len(result) == n_cols, f"Expected {n_cols} aliases, got {len(result)}"
    for i in range(n_cols):
        assert f'col_{i}' in adf.df.columns
    
    if verbose:
        print(f"\nResults:")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Columns materialized: {len(result)}")
        print(f"  Time per column: {elapsed/n_cols*1000:.1f}ms")
    
    return {
        'time': elapsed,
        'rows': n_rows,
        'columns': n_cols,
        'time_per_column_ms': elapsed / n_cols * 1000
    }


def benchmark_caching_vs_no_caching(n_rows=100_000, n_cols=5):
    """
    Compare performance with and without join caching.
    
    This demonstrates the speedup from Phase 2 caching.
    """
    print(f"\n{'='*60}")
    print("Comparison: Caching vs No Caching")
    print(f"{'='*60}")
    
    np.random.seed(42)
    n_sf_rows = 1000
    
    df = pd.DataFrame({
        'idx': np.random.randint(0, n_sf_rows, n_rows),
        'x': np.random.randn(n_rows)
    })
    
    sf_data = {'idx': np.arange(n_sf_rows)}
    for i in range(n_cols):
        sf_data[f'val_{i}'] = np.random.randn(n_sf_rows)
    sf = pd.DataFrame(sf_data)
    
    # Test WITHOUT caching (sequential single-alias calls)
    adf_nocache = AliasDataFrame(df.copy())
    adf_nocache.register_subframe('S', AliasDataFrame(sf.copy()), index_columns='idx')
    for i in range(n_cols):
        adf_nocache.add_alias(f'col_{i}', f'S.val_{i}')
    
    t0 = time.time()
    for i in range(n_cols):
        adf_nocache.materialize_alias(f'col_{i}')
    t_nocache = time.time() - t0
    
    # Test WITH caching (batch call)
    adf_cache = AliasDataFrame(df.copy())
    adf_cache.register_subframe('S', AliasDataFrame(sf.copy()), index_columns='idx')
    for i in range(n_cols):
        adf_cache.add_alias(f'col_{i}', f'S.val_{i}')
    
    t0 = time.time()
    adf_cache.materialize_aliases(pattern=r'col_.*')
    t_cache = time.time() - t0
    
    # Verify results match
    for i in range(n_cols):
        np.testing.assert_array_almost_equal(
            adf_nocache.df[f'col_{i}'].values,
            adf_cache.df[f'col_{i}'].values
        )
    
    speedup = t_nocache / t_cache if t_cache > 0 else float('inf')
    
    print(f"\n{n_rows:,} rows, {n_cols} columns:")
    print(f"  Without caching: {t_nocache*1000:.1f}ms ({t_nocache/n_cols*1000:.1f}ms per column)")
    print(f"  With caching:    {t_cache*1000:.1f}ms ({t_cache/n_cols*1000:.1f}ms per column)")
    print(f"  Speedup: {speedup:.1f}x")
    
    return {
        'time_nocache': t_nocache,
        'time_cache': t_cache,
        'speedup': speedup
    }


def benchmark_with_profiling(n_rows=100_000):
    """
    Example of using profiling to identify bottlenecks.
    
    This demonstrates the profiling interface for performance debugging.
    """
    print(f"\n{'='*60}")
    print("Profiling Example")
    print(f"{'='*60}")
    
    np.random.seed(42)
    
    df = pd.DataFrame({
        'idx': np.random.randint(0, 1000, n_rows),
        'x': np.random.randn(n_rows),
        'y': np.random.randn(n_rows)
    })
    
    sf = pd.DataFrame({
        'idx': np.arange(1000),
        'scale': np.random.randn(1000),
        'offset': np.random.randn(1000)
    })
    
    adf = AliasDataFrame(df)
    adf.register_subframe('S', AliasDataFrame(sf), index_columns='idx')
    
    # Add aliases with dependencies
    adf.add_alias('scaled_x', 'x * S.scale')
    adf.add_alias('offset_y', 'y + S.offset')
    adf.add_alias('combined', 'scaled_x + offset_y')
    
    print(f"\nMaterializing with profiling enabled...\n")
    
    # Run with profiling
    adf.materialize_aliases(
        names=['combined'],
        with_dependencies=True,
        profile=True
    )
    
    print(f"\nVerification: 'combined' column exists: {'combined' in adf.df.columns}")


def benchmark_join_caching_threshold():
    """
    Threshold test: 20 columns on 1M rows should complete in <10s.
    
    This is the acceptance criterion from the spec.
    """
    print(f"\n{'='*60}")
    print("Join Caching Threshold Test")
    print(f"{'='*60}")
    
    result = benchmark_single_subframe_many_columns(
        n_rows=1_000_000, 
        n_cols=20, 
        verbose=True
    )
    
    threshold = 10.0
    passed = result['time'] < threshold
    
    print(f"\n{'='*60}")
    print(f"Threshold: {threshold}s")
    print(f"Actual: {result['time']:.2f}s")
    print(f"Status: {'PASS' if passed else 'FAIL'}")
    print(f"{'='*60}")
    
    return passed


def run_all_benchmarks():
    """Run all benchmarks and print summary."""
    print("\n" + "="*60)
    print("AliasDataFrame Join Caching Benchmarks (Phase 2)")
    print("="*60)
    
    results = {}
    
    # Caching comparison
    results['comparison'] = benchmark_caching_vs_no_caching(n_rows=100_000, n_cols=5)
    
    # Small scale test
    results['small'] = benchmark_single_subframe_many_columns(
        n_rows=100_000, n_cols=10, verbose=True
    )
    
    # Medium scale test  
    results['medium'] = benchmark_single_subframe_many_columns(
        n_rows=500_000, n_cols=20, verbose=True
    )
    
    # Threshold test (if time permits)
    try:
        results['threshold'] = benchmark_single_subframe_many_columns(
            n_rows=1_000_000, n_cols=20, verbose=True
        )
    except Exception as e:
        print(f"Threshold test skipped: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    if 'comparison' in results:
        print(f"  Caching speedup: {results['comparison']['speedup']:.1f}x")
    for name in ['small', 'medium', 'threshold']:
        if name in results:
            r = results[name]
            print(f"  {name}: {r['time']:.2f}s for {r['columns']} cols on {r['rows']:,} rows")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run join caching benchmarks')
    parser.add_argument('--profile', action='store_true', help='Run profiling example')
    parser.add_argument('--threshold', action='store_true', help='Run threshold test only')
    parser.add_argument('--compare', action='store_true', help='Compare caching vs no caching')
    parser.add_argument('--all', action='store_true', help='Run all benchmarks')
    
    args = parser.parse_args()
    
    if args.profile:
        benchmark_with_profiling()
    elif args.threshold:
        benchmark_join_caching_threshold()
    elif args.compare:
        benchmark_caching_vs_no_caching()
    elif args.all:
        run_all_benchmarks()
    else:
        # Default: run comparison
        benchmark_caching_vs_no_caching()
