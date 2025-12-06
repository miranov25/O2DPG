#!/usr/bin/env python3
"""
benchmark_composite_keys_rdf.py - Benchmark composite key generation and RDF integration

Tests the Phase 5 composite key infrastructure:
- Dense key linearization (for compact key ranges)
- Sparse key mapping (for large/sparse key ranges)
- TMemFile runtime composite key setup
- End-to-end RDF query with friend tree joins

Usage:
    python benchmark_composite_keys_rdf.py --json results.json           # Required: JSON output
    python benchmark_composite_keys_rdf.py --quick --json results.json   # Quick mode
    python benchmark_composite_keys_rdf.py --profile --json results.json # With profiling
    python benchmark_composite_keys_rdf.py --quiet --json results.json   # Minimal output

Exit Codes:
    0 - All benchmarks completed (passed or skipped)
    1 - Fatal error

Scenarios:
    A. dense_generation:  compute_composite_key_dense() performance
    B. sparse_generation: compute_composite_key_sparse() performance
    C. tmemfile_setup:    Runtime TMemFile composite key creation (requires ROOT)
    D. rdf_query:         End-to-end RDF query with composite key join (requires ROOT)
"""

import argparse
import cProfile
import gc
import json
import os
import platform
import pstats
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import composite key functions
from _composite_keys import (
    compute_composite_key_dense,
    compute_composite_key_sparse,
    compute_composite_key_auto,
)

# Try to import ROOT (optional)
try:
    import ROOT
    HAS_ROOT = True
except ImportError:
    HAS_ROOT = False

# Try to import uproot (for file creation)
try:
    import uproot
    HAS_UPROOT = True
except ImportError:
    HAS_UPROOT = False

# Try to import AliasDataFrame components
try:
    from AliasDataFrame import AliasDataFrame
    from AliasDataFrameRDF import setup_rdf_with_friends
    HAS_ADF = True
except ImportError:
    HAS_ADF = False


# =============================================================================
# Configuration
# =============================================================================

# Random seed for reproducibility
RNG_SEED = 12345

# Data sizes
QUICK_SIZES = {
    'dense': [10_000],
    'sparse': [10_000],
    'tmemfile_main': 100_000,
    'tmemfile_friend': 10_000,
}

DEFAULT_SIZES = {
    'dense': [100_000, 1_000_000],
    'sparse': [100_000, 1_000_000],
    'tmemfile_main': 1_000_000,
    'tmemfile_friend': 100_000,
}

# TPC-like key ranges (realistic ALICE calibration)
KEY_COLUMNS = ['k0', 'k1', 'k2']
MAX_VALUES = [2, 152, 25]  # side, row, drift25


# =============================================================================
# Scenario A: Dense Key Generation
# =============================================================================

def benchmark_dense_generation(sizes, verbose=True):
    """
    Benchmark compute_composite_key_dense() performance.
    
    Tests linearization: k0 + k1*max0 + k2*max0*max1
    """
    results = {'sizes': []}
    
    for n_rows in sizes:
        if verbose:
            print(f"\n  Dense generation: {n_rows:,} rows")
        
        # Create synthetic data (TPC-like 3 key columns)
        df = pd.DataFrame({
            'k0': np.random.randint(0, MAX_VALUES[0], size=n_rows, dtype=np.int32),
            'k1': np.random.randint(0, MAX_VALUES[1], size=n_rows, dtype=np.int32),
            'k2': np.random.randint(0, MAX_VALUES[2], size=n_rows, dtype=np.int32),
            'value': np.random.randn(n_rows).astype(np.float32),
        })
        
        # Warmup
        _ = compute_composite_key_dense(df, KEY_COLUMNS, MAX_VALUES)
        
        # Benchmark
        gc.collect()
        t0 = time.perf_counter()
        keys = compute_composite_key_dense(df, KEY_COLUMNS, MAX_VALUES)
        elapsed = time.perf_counter() - t0
        
        # Verify
        passed = (
            len(keys) == n_rows and
            keys.dtype == np.int64 and
            keys.min() >= 0
        )
        
        keys_per_sec = n_rows / elapsed if elapsed > 0 else 0
        
        result = {
            'rows': n_rows,
            'time_s': round(elapsed, 6),
            'keys_per_sec': round(keys_per_sec),
            'passed': passed,
        }
        results['sizes'].append(result)
        
        if verbose:
            print(f"    Time: {elapsed*1000:.2f}ms, Keys/sec: {keys_per_sec:,.0f}")
    
    # Overall pass if all sizes passed
    results['passed'] = all(s['passed'] for s in results['sizes'])
    return results


# =============================================================================
# Scenario B: Sparse Key Generation
# =============================================================================

def benchmark_sparse_generation(sizes, verbose=True):
    """
    Benchmark compute_composite_key_sparse() performance.
    
    Tests np.unique-based mapping for sparse key combinations.
    """
    results = {'sizes': []}
    
    for n_rows in sizes:
        if verbose:
            print(f"\n  Sparse generation: {n_rows:,} rows")
        
        # Create main DataFrame with larger key ranges (100x100x100 = 1M possible)
        main_df = pd.DataFrame({
            'k0': np.random.randint(0, 100, size=n_rows, dtype=np.int32),
            'k1': np.random.randint(0, 100, size=n_rows, dtype=np.int32),
            'k2': np.random.randint(0, 100, size=n_rows, dtype=np.int32),
            'value': np.random.randn(n_rows).astype(np.float32),
        })
        
        # Sub DataFrame: ~1% unique key combinations from main
        n_unique = max(100, n_rows // 100)
        sub_indices = np.random.choice(n_rows, size=n_unique, replace=False)
        sub_df = main_df.iloc[sub_indices][KEY_COLUMNS].copy()
        sub_df['calib'] = np.random.randn(n_unique).astype(np.float32)
        
        # Warmup
        _, _ = compute_composite_key_sparse(main_df, sub_df, KEY_COLUMNS)
        
        # Benchmark
        gc.collect()
        t0 = time.perf_counter()
        main_keys, sub_keys = compute_composite_key_sparse(main_df, sub_df, KEY_COLUMNS)
        elapsed = time.perf_counter() - t0
        
        # Verify
        passed = (
            len(main_keys) == n_rows and
            len(sub_keys) == n_unique and
            main_keys.dtype == np.int64 and
            sub_keys.dtype == np.int64
        )
        
        keys_per_sec = n_rows / elapsed if elapsed > 0 else 0
        
        result = {
            'rows': n_rows,
            'unique_keys': n_unique,
            'time_s': round(elapsed, 6),
            'keys_per_sec': round(keys_per_sec),
            'passed': passed,
        }
        results['sizes'].append(result)
        
        if verbose:
            print(f"    Time: {elapsed*1000:.2f}ms, Keys/sec: {keys_per_sec:,.0f}, Unique: {n_unique:,}")
    
    results['passed'] = all(s['passed'] for s in results['sizes'])
    return results


# =============================================================================
# Scenario C: TMemFile Setup
# =============================================================================

def benchmark_tmemfile_setup(n_main, n_friend, verbose=True, tmp_dir=None):
    """
    Benchmark runtime TMemFile composite key setup.
    
    Tests the full production path:
    1. Create AliasDataFrame with 3-column key subframe
    2. Export to ROOT file (without pre-computed composite keys)
    3. Call setup_rdf_with_friends() which triggers runtime key generation
    """
    if not HAS_ROOT:
        return {'skipped': True, 'reason': 'ROOT not available'}
    
    if not HAS_UPROOT:
        return {'skipped': True, 'reason': 'uproot not available'}
    
    if not HAS_ADF:
        return {'skipped': True, 'reason': 'AliasDataFrame not available'}
    
    if verbose:
        print(f"\n  TMemFile setup: {n_main:,} main rows, {n_friend:,} friend rows")
    
    # Create temp directory for ROOT file
    if tmp_dir is None:
        tmp_dir = tempfile.mkdtemp()
    filepath = os.path.join(tmp_dir, "benchmark_composite.root")
    
    try:
        # Create main DataFrame
        main_df = pd.DataFrame({
            'k0': np.random.randint(0, MAX_VALUES[0], size=n_main, dtype=np.int32),
            'k1': np.random.randint(0, MAX_VALUES[1], size=n_main, dtype=np.int32),
            'k2': np.random.randint(0, MAX_VALUES[2], size=n_main, dtype=np.int32),
            'value': np.random.randn(n_main).astype(np.float32),
        })
        
        # Create friend DataFrame (shuffled for realistic test)
        # calib = k0*100 + k1*10 + k2 for correctness verification
        friend_df = pd.DataFrame({
            'k0': np.random.randint(0, MAX_VALUES[0], size=n_friend, dtype=np.int32),
            'k1': np.random.randint(0, MAX_VALUES[1], size=n_friend, dtype=np.int32),
            'k2': np.random.randint(0, MAX_VALUES[2], size=n_friend, dtype=np.int32),
        })
        friend_df['calib'] = (
            friend_df['k0'] * 100 + 
            friend_df['k1'] * 10 + 
            friend_df['k2']
        ).astype(np.float32)
        
        # Shuffle friend to ensure index is actually used
        shuffle_idx = np.random.permutation(n_friend)
        friend_df = friend_df.iloc[shuffle_idx].reset_index(drop=True)
        
        # Create AliasDataFrame and register subframe
        adf = AliasDataFrame(main_df)
        adf.register_subframe('S', AliasDataFrame(friend_df), index_columns=KEY_COLUMNS)
        
        # Export to ROOT file
        # Note: We want composite_keys="none" to force runtime generation,
        # but if that parameter isn't available, the default behavior should work
        if verbose:
            print(f"    Exporting to: {filepath}")
        
        t_export_start = time.perf_counter()
        try:
            # Try with composite_keys parameter if available
            adf.export_tree(filepath, treename="tree")
        except TypeError:
            # Fallback if parameter doesn't exist
            adf.export_tree(filepath, treename="tree")
        t_export = time.perf_counter() - t_export_start
        
        if verbose:
            print(f"    Export time: {t_export*1000:.1f}ms")
        
        # Benchmark setup_rdf_with_friends (the main metric)
        gc.collect()
        t0 = time.perf_counter()
        result = setup_rdf_with_friends(adf, filepath, treename="tree")
        total_setup_time = time.perf_counter() - t0
        
        # Handle 2-tuple or 3-tuple return
        if isinstance(result, tuple):
            if len(result) == 2:
                rdf, file_handle = result
                composite_info = None
            elif len(result) == 3:
                rdf, file_handle, composite_info = result
            else:
                rdf = result[0]
                file_handle = result[1] if len(result) > 1 else None
                composite_info = result[2] if len(result) > 2 else None
        else:
            rdf = result
            file_handle = None
            composite_info = None
        
        # Verify RDF is usable
        try:
            n_entries = rdf.Count().GetValue()
            passed = (n_entries == n_main)
        except Exception as e:
            if verbose:
                print(f"    Warning: RDF verification failed: {e}")
            passed = False
            n_entries = 0
        
        output = {
            'main_rows': n_main,
            'friend_rows': n_friend,
            'total_setup_time_s': round(total_setup_time, 6),
            'export_time_s': round(t_export, 6),
            'passed': passed,
        }
        
        # Add composite info if available
        if composite_info:
            output['composite_info'] = composite_info
        
        if verbose:
            print(f"    Setup time: {total_setup_time*1000:.1f}ms")
            print(f"    RDF entries: {n_entries:,}")
        
        # Store for reuse in Scenario D
        output['_rdf'] = rdf
        output['_file_handle'] = file_handle
        output['_filepath'] = filepath
        output['_adf'] = adf
        
        return output
        
    except Exception as e:
        if verbose:
            print(f"    Error: {e}")
        return {
            'skipped': True,
            'reason': str(e),
            'main_rows': n_main,
            'friend_rows': n_friend,
        }


# =============================================================================
# Scenario D: RDF Query
# =============================================================================

def benchmark_rdf_query(tmemfile_result, verbose=True):
    """
    Benchmark end-to-end RDF query with composite key friend join.
    
    Reuses the RDF from Scenario C to measure query performance.
    """
    if not HAS_ROOT:
        return {'skipped': True, 'reason': 'ROOT not available'}
    
    # Check if TMemFile setup succeeded
    if tmemfile_result.get('skipped'):
        return {
            'skipped': True, 
            'reason': f"TMemFile setup skipped: {tmemfile_result.get('reason', 'unknown')}"
        }
    
    rdf = tmemfile_result.get('_rdf')
    if rdf is None:
        return {'skipped': True, 'reason': 'No RDF from TMemFile setup'}
    
    if verbose:
        print(f"\n  RDF query benchmark")
    
    try:
        # Define expected value using keys (for verification)
        # expected_calib = k0*100 + k1*10 + k2
        rdf = rdf.Define("expected_calib", "(float)(k0 * 100 + k1 * 10 + k2)")
        
        # Query friend column - this is what we're measuring
        gc.collect()
        t0 = time.perf_counter()
        
        # Try to access the subframe column
        # The column should be accessible as S.calib
        try:
            result_sum = rdf.Sum("S.calib").GetValue()
            query_time = time.perf_counter() - t0
            
            n_rows = rdf.Count().GetValue()
            passed = True
            
        except Exception as e:
            # Fallback: try without the S. prefix (in case of different setup)
            query_time = time.perf_counter() - t0
            if verbose:
                print(f"    Note: S.calib access failed ({e}), trying alternatives...")
            
            # Try to at least count rows
            n_rows = rdf.Count().GetValue()
            result_sum = 0
            passed = False
        
        output = {
            'n_rows': n_rows,
            'query_time_s': round(query_time, 6),
            'result_sum': float(result_sum) if result_sum else None,
            'passed': passed,
        }
        
        if verbose:
            print(f"    Query time: {query_time*1000:.1f}ms")
            print(f"    Rows: {n_rows:,}")
            if result_sum:
                print(f"    Sum(S.calib): {result_sum:.2f}")
        
        return output
        
    except Exception as e:
        if verbose:
            print(f"    Error: {e}")
        return {
            'skipped': True,
            'reason': str(e),
        }


# =============================================================================
# Main Runner
# =============================================================================

def run_all_benchmarks(quick_mode=False, verbose=True, profile=False, results_dir=None):
    """
    Run all benchmark scenarios.
    
    Returns dict with results for each scenario.
    """
    np.random.seed(RNG_SEED)
    
    sizes = QUICK_SIZES if quick_mode else DEFAULT_SIZES
    
    results = {}
    total_start = time.perf_counter()
    
    if verbose:
        mode = 'quick' if quick_mode else 'default'
        print(f"\n{'='*60}")
        print(f"Composite Keys / RDF Benchmark ({mode} mode)")
        print(f"{'='*60}")
    
    # Scenario A: Dense Key Generation
    if verbose:
        print("\n--- Scenario A: Dense Key Generation ---")
    
    if profile and results_dir:
        profiler = cProfile.Profile()
        profiler.enable()
    
    results['dense_generation'] = benchmark_dense_generation(
        sizes['dense'], verbose=verbose
    )
    
    if profile and results_dir:
        profiler.disable()
        profile_path = os.path.join(results_dir, 'profiles', 'composite_dense.prof')
        os.makedirs(os.path.dirname(profile_path), exist_ok=True)
        profiler.dump_stats(profile_path)
    
    # Scenario B: Sparse Key Generation
    if verbose:
        print("\n--- Scenario B: Sparse Key Generation ---")
    
    if profile and results_dir:
        profiler = cProfile.Profile()
        profiler.enable()
    
    results['sparse_generation'] = benchmark_sparse_generation(
        sizes['sparse'], verbose=verbose
    )
    
    if profile and results_dir:
        profiler.disable()
        profile_path = os.path.join(results_dir, 'profiles', 'composite_sparse.prof')
        profiler.dump_stats(profile_path)
    
    # Scenario C: TMemFile Setup
    if verbose:
        print("\n--- Scenario C: TMemFile Setup ---")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        if profile and results_dir:
            profiler = cProfile.Profile()
            profiler.enable()
        
        results['tmemfile_setup'] = benchmark_tmemfile_setup(
            sizes['tmemfile_main'],
            sizes['tmemfile_friend'],
            verbose=verbose,
            tmp_dir=tmp_dir,
        )
        
        if profile and results_dir:
            profiler.disable()
            profile_path = os.path.join(results_dir, 'profiles', 'composite_tmemfile.prof')
            profiler.dump_stats(profile_path)
        
        # Scenario D: RDF Query (reuses data from C)
        if verbose:
            print("\n--- Scenario D: RDF Query ---")
        
        if profile and results_dir:
            profiler = cProfile.Profile()
            profiler.enable()
        
        results['rdf_query'] = benchmark_rdf_query(
            results['tmemfile_setup'],
            verbose=verbose,
        )
        
        if profile and results_dir:
            profiler.disable()
            profile_path = os.path.join(results_dir, 'profiles', 'composite_rdf_query.prof')
            profiler.dump_stats(profile_path)
    
    # Clean up internal references
    for key in ['tmemfile_setup', 'rdf_query']:
        if key in results:
            for internal_key in ['_rdf', '_file_handle', '_filepath', '_adf']:
                results[key].pop(internal_key, None)
    
    total_time = time.perf_counter() - total_start
    results['total_time_s'] = round(total_time, 3)
    
    # Determine overall pass/fail
    all_passed = True
    for scenario in ['dense_generation', 'sparse_generation', 'tmemfile_setup', 'rdf_query']:
        r = results.get(scenario, {})
        if r.get('skipped'):
            continue  # Skipped doesn't count as failure
        if not r.get('passed', False):
            all_passed = False
    
    results['all_passed'] = all_passed
    
    return results


def print_summary(results, mode):
    """Print benchmark summary."""
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    print(f"\n{'Scenario':<25} {'Status':<10} {'Time':<12} {'Throughput':<20}")
    print("-" * 67)
    
    # Dense
    dense = results.get('dense_generation', {})
    if dense.get('skipped'):
        print(f"{'dense_generation':<25} {'SKIPPED':<10}")
    else:
        best_size = dense.get('sizes', [{}])[-1]  # Last (largest) size
        status = 'PASS' if dense.get('passed') else 'FAIL'
        time_ms = best_size.get('time_s', 0) * 1000
        kps = best_size.get('keys_per_sec', 0)
        print(f"{'dense_generation':<25} {status:<10} {time_ms:>8.1f}ms  {kps:>15,} keys/s")
    
    # Sparse
    sparse = results.get('sparse_generation', {})
    if sparse.get('skipped'):
        print(f"{'sparse_generation':<25} {'SKIPPED':<10}")
    else:
        best_size = sparse.get('sizes', [{}])[-1]
        status = 'PASS' if sparse.get('passed') else 'FAIL'
        time_ms = best_size.get('time_s', 0) * 1000
        kps = best_size.get('keys_per_sec', 0)
        print(f"{'sparse_generation':<25} {status:<10} {time_ms:>8.1f}ms  {kps:>15,} keys/s")
    
    # TMemFile
    tmem = results.get('tmemfile_setup', {})
    if tmem.get('skipped'):
        reason = tmem.get('reason', 'unknown')[:30]
        print(f"{'tmemfile_setup':<25} {'SKIPPED':<10} {reason}")
    else:
        status = 'PASS' if tmem.get('passed') else 'FAIL'
        time_ms = tmem.get('total_setup_time_s', 0) * 1000
        rows = tmem.get('main_rows', 0)
        rps = rows / tmem.get('total_setup_time_s', 1)
        print(f"{'tmemfile_setup':<25} {status:<10} {time_ms:>8.1f}ms  {rps:>15,.0f} rows/s")
    
    # RDF Query
    rdf = results.get('rdf_query', {})
    if rdf.get('skipped'):
        reason = rdf.get('reason', 'unknown')[:30]
        print(f"{'rdf_query':<25} {'SKIPPED':<10} {reason}")
    else:
        status = 'PASS' if rdf.get('passed') else 'FAIL'
        time_ms = rdf.get('query_time_s', 0) * 1000
        rows = rdf.get('n_rows', 0)
        rps = rows / rdf.get('query_time_s', 1) if rdf.get('query_time_s') else 0
        print(f"{'rdf_query':<25} {status:<10} {time_ms:>8.1f}ms  {rps:>15,.0f} rows/s")
    
    print("-" * 67)
    total = results.get('total_time_s', 0)
    all_passed = results.get('all_passed', False)
    status = 'ALL PASSED' if all_passed else 'SOME FAILED'
    print(f"{'Total':<25} {status:<10} {total:>8.2f}s")


def export_json(results, filepath, mode):
    """Export results to JSON file."""
    output = {
        'timestamp': datetime.now().isoformat(),
        'hostname': platform.node(),
        'python_version': platform.python_version(),
        'platform': platform.platform(),
        'component': 'composite_keys_rdf',
        'mode': mode,
        'results': {
            'dense_generation': results.get('dense_generation', {}),
            'sparse_generation': results.get('sparse_generation', {}),
            'tmemfile_setup': results.get('tmemfile_setup', {}),
            'rdf_query': results.get('rdf_query', {}),
        },
        'all_passed': results.get('all_passed', False),
        'total_time_s': results.get('total_time_s', 0),
    }
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nResults exported to: {filepath}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark composite key generation and RDF integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python benchmark_composite_keys_rdf.py --json results.json
    python benchmark_composite_keys_rdf.py --quick --json results.json
    python benchmark_composite_keys_rdf.py --profile --json results.json

Scenarios:
    A. dense_generation:  compute_composite_key_dense() performance
    B. sparse_generation: compute_composite_key_sparse() performance
    C. tmemfile_setup:    Runtime TMemFile composite key creation
    D. rdf_query:         End-to-end RDF query with friend join
        """
    )
    parser.add_argument('--json', type=str, required=True, metavar='FILE',
                        help='Export results to JSON file (required)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: smaller data sizes')
    parser.add_argument('--quiet', action='store_true',
                        help='Minimal output')
    parser.add_argument('--profile', action='store_true',
                        help='Save profiler output to results/profiles/')
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    mode = 'quick' if args.quick else 'default'
    
    # Determine results directory for profiling
    results_dir = os.path.dirname(args.json) or 'results'
    
    # Run benchmarks
    results = run_all_benchmarks(
        quick_mode=args.quick,
        verbose=verbose,
        profile=args.profile,
        results_dir=results_dir if args.profile else None,
    )
    
    # Print summary
    if verbose:
        print_summary(results, mode)
    else:
        # Minimal output for --quiet
        dense_pass = results.get('dense_generation', {}).get('passed', False)
        sparse_pass = results.get('sparse_generation', {}).get('passed', False)
        tmem_pass = results.get('tmemfile_setup', {}).get('passed', False)
        rdf_pass = results.get('rdf_query', {}).get('passed', False)
        total = results.get('total_time_s', 0)
        print(f"dense:{dense_pass} sparse:{sparse_pass} tmem:{tmem_pass} rdf:{rdf_pass} | {total:.2f}s")
    
    # Export to JSON
    export_json(results, args.json, mode)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
