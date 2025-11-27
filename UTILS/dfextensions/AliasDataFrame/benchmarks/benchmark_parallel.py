#!/usr/bin/env python3
"""
benchmark_parallel.py - Test read_tree scaling with num_workers

Identifies optimal worker count and detects parallel reading issues.

IMPORTANT: This benchmark uses signal.SIGALRM for timeout detection,
which is POSIX-only (Linux, macOS). It will NOT work on Windows.

Usage:
    python benchmark_parallel.py <input_file.root>
    python benchmark_parallel.py data.root --max-workers 8
    python benchmark_parallel.py data.root --timeout 30 --repeats 5
    python benchmark_parallel.py data.root --json results.json

Exit Codes:
    0 - Always (issues are reported, not fatal)
"""

import argparse
import gc
import json
import os
import platform
import signal
import sys
import time
import tracemalloc
from datetime import datetime

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AliasDataFrame import AliasDataFrame


# =============================================================================
# Configuration
# =============================================================================

# Default worker counts to test
DEFAULT_WORKER_COUNTS = [1, 2, 4, 8]

# Timeout for detecting hangs
DEFAULT_TIMEOUT = 60  # seconds

# Number of repetitions for statistical validity
DEFAULT_REPEATS = 3

# Variance threshold for detecting instability (std/mean)
VARIANCE_THRESHOLD = 0.5


# =============================================================================
# Timeout Handling (POSIX only)
# =============================================================================

class TimeoutError(Exception):
    """Raised when operation times out."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for SIGALRM."""
    raise TimeoutError("Operation timed out")


def is_posix():
    """Check if running on POSIX system."""
    return hasattr(signal, 'SIGALRM')


# =============================================================================
# Benchmark Functions
# =============================================================================

def benchmark_single_read(filepath, treename, num_workers, timeout):
    """
    Benchmark a single read_tree call with timeout.
    
    Parameters
    ----------
    filepath : str
        Path to ROOT file
    treename : str
        Name of tree in file
    num_workers : int
        Number of parallel workers
    timeout : int
        Timeout in seconds
        
    Returns
    -------
    dict : {time_s, rows, memory_mb, status}
    """
    result = {
        'time_s': None,
        'rows': None,
        'memory_mb': None,
        'status': 'unknown'
    }
    
    # Set up timeout (POSIX only)
    if is_posix():
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
    
    try:
        gc.collect()
        tracemalloc.start()
        
        t0 = time.perf_counter()
        adf = AliasDataFrame.read_tree(filepath, treename, num_workers=num_workers)
        elapsed = time.perf_counter() - t0
        
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        if is_posix():
            signal.alarm(0)  # Cancel alarm
        
        result['time_s'] = elapsed
        result['rows'] = len(adf.df)
        result['memory_mb'] = peak / (1024 * 1024)
        result['status'] = 'ok'
        
        del adf
        gc.collect()
        
    except TimeoutError:
        if is_posix():
            signal.alarm(0)
        tracemalloc.stop()
        result['status'] = 'timeout'
        
    except Exception as e:
        if is_posix():
            signal.alarm(0)
        tracemalloc.stop()
        result['status'] = f'error: {str(e)}'
    
    return result


def benchmark_workers(filepath, treename="tree", worker_counts=None, 
                      timeout=DEFAULT_TIMEOUT, repeats=DEFAULT_REPEATS,
                      verbose=True):
    """
    Benchmark read_tree with different num_workers values.
    
    Parameters
    ----------
    filepath : str
        Path to ROOT file
    treename : str
        Name of tree in file
    worker_counts : list of int
        Worker counts to test
    timeout : int
        Timeout per read attempt (seconds)
    repeats : int
        Number of repetitions per worker count
    verbose : bool
        Print progress
        
    Returns
    -------
    dict : Results for each worker count
    """
    if worker_counts is None:
        worker_counts = DEFAULT_WORKER_COUNTS
    
    results = {}
    
    if verbose:
        print("=" * 70)
        print("PARALLEL READ BENCHMARK")
        print("=" * 70)
        print(f"File:      {filepath}")
        print(f"Tree:      {treename}")
        print(f"Workers:   {worker_counts}")
        print(f"Repeats:   {repeats}")
        print(f"Timeout:   {timeout}s")
        print(f"Hostname:  {platform.node()}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        
        if not is_posix():
            print("\n⚠️  WARNING: Not running on POSIX system.")
            print("   Timeout detection is disabled. Hangs will block indefinitely.")
        print()
    
    for n_workers in worker_counts:
        if verbose:
            print(f"Testing num_workers={n_workers}...")
        
        times = []
        memories = []
        failures = 0
        rows = None
        
        for i in range(repeats):
            result = benchmark_single_read(filepath, treename, n_workers, timeout)
            
            if result['status'] == 'ok':
                times.append(result['time_s'])
                memories.append(result['memory_mb'])
                rows = result['rows']
                if verbose:
                    print(f"  Run {i+1}: {result['time_s']:.2f}s "
                          f"({result['rows']:,} rows, {result['memory_mb']:.1f} MB)")
            else:
                failures += 1
                if verbose:
                    print(f"  Run {i+1}: {result['status'].upper()}")
            
            # Brief pause between runs
            time.sleep(0.5)
        
        # Compute statistics
        results[n_workers] = {
            'times': times,
            'mean_s': np.mean(times) if times else None,
            'std_s': np.std(times) if times else None,
            'min_s': np.min(times) if times else None,
            'max_s': np.max(times) if times else None,
            'memory_mb': np.mean(memories) if memories else None,
            'rows': rows,
            'failures': failures,
            'total_runs': repeats
        }
        
        if verbose and times:
            mean = results[n_workers]['mean_s']
            std = results[n_workers]['std_s']
            print(f"  → Mean: {mean:.2f}s ± {std:.2f}s")
        if verbose and failures:
            print(f"  → Failures: {failures}/{repeats}")
        if verbose:
            print()
    
    return results


def analyze_results(results, verbose=True):
    """
    Analyze benchmark results for issues and recommendations.
    
    Returns
    -------
    dict : Analysis with 'optimal', 'issues', 'recommendations'
    """
    analysis = {
        'optimal': None,
        'issues': [],
        'recommendations': []
    }
    
    # Find optimal worker count
    valid = {k: v for k, v in results.items() 
             if v['mean_s'] is not None and v['failures'] == 0}
    
    if valid:
        optimal = min(valid, key=lambda k: valid[k]['mean_s'])
        analysis['optimal'] = {
            'num_workers': optimal,
            'mean_s': valid[optimal]['mean_s'],
            'std_s': valid[optimal]['std_s']
        }
    
    # Detect issues
    for n_workers, data in results.items():
        # Issue: Timeouts or failures
        if data['failures'] > 0:
            analysis['issues'].append({
                'type': 'failures',
                'num_workers': n_workers,
                'count': data['failures'],
                'message': f"num_workers={n_workers}: {data['failures']} timeouts/failures"
            })
        
        # Issue: High variance (unstable)
        if data['times'] and len(data['times']) > 1:
            cv = data['std_s'] / data['mean_s'] if data['mean_s'] > 0 else 0
            if cv > VARIANCE_THRESHOLD:
                analysis['issues'].append({
                    'type': 'variance',
                    'num_workers': n_workers,
                    'cv': cv,
                    'message': f"num_workers={n_workers}: high variance (CV={cv:.2f})"
                })
        
        # Issue: Negative scaling (more workers = slower)
        if analysis['optimal'] and n_workers > analysis['optimal']['num_workers']:
            if data['mean_s'] and data['mean_s'] > analysis['optimal']['mean_s'] * 1.2:
                analysis['issues'].append({
                    'type': 'negative_scaling',
                    'num_workers': n_workers,
                    'message': f"num_workers={n_workers}: slower than optimal "
                               f"({data['mean_s']:.2f}s vs {analysis['optimal']['mean_s']:.2f}s)"
                })
    
    # Generate recommendations
    if analysis['issues']:
        # Check for failures with high worker counts
        failure_workers = [i['num_workers'] for i in analysis['issues'] if i['type'] == 'failures']
        if failure_workers:
            max_safe = min(failure_workers) - 1
            if max_safe >= 1:
                analysis['recommendations'].append(
                    f"Consider using num_workers<={max_safe} to avoid timeouts"
                )
        
        # Check for high variance
        variance_workers = [i['num_workers'] for i in analysis['issues'] if i['type'] == 'variance']
        if variance_workers:
            analysis['recommendations'].append(
                "High variance detected - server may be under load or I/O is inconsistent"
            )
    
    if analysis['optimal']:
        analysis['recommendations'].append(
            f"Recommended: num_workers={analysis['optimal']['num_workers']} "
            f"({analysis['optimal']['mean_s']:.2f}s average)"
        )
    
    return analysis


def print_summary(results, analysis):
    """Print summary of benchmark results."""
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Workers':<10} {'Mean (s)':<12} {'Std (s)':<12} {'Min (s)':<12} {'Failures':<10}")
    print("-" * 70)
    
    for n_workers, data in sorted(results.items()):
        mean = f"{data['mean_s']:.2f}" if data['mean_s'] else "N/A"
        std = f"{data['std_s']:.2f}" if data['std_s'] else "N/A"
        min_t = f"{data['min_s']:.2f}" if data['min_s'] else "N/A"
        failures = f"{data['failures']}/{data['total_runs']}"
        
        marker = " *" if (analysis['optimal'] and 
                         n_workers == analysis['optimal']['num_workers']) else ""
        print(f"{n_workers:<10} {mean:<12} {std:<12} {min_t:<12} {failures:<10}{marker}")
    
    if analysis['optimal']:
        print(f"\n* Optimal: num_workers={analysis['optimal']['num_workers']}")
    
    # Issues
    print("\n" + "=" * 70)
    print("ISSUES DETECTED")
    print("=" * 70)
    
    if analysis['issues']:
        for issue in analysis['issues']:
            print(f"  ⚠ {issue['message']}")
    else:
        print("  ✓ No issues detected")
    
    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    
    for rec in analysis['recommendations']:
        print(f"  → {rec}")
    
    print("=" * 70)


def export_json(results, analysis, filepath, input_file, treename):
    """Export results to JSON file."""
    output = {
        'timestamp': datetime.now().isoformat(),
        'hostname': platform.node(),
        'python_version': platform.python_version(),
        'platform': platform.platform(),
        'input_file': input_file,
        'treename': treename,
        'results': {str(k): v for k, v in results.items()},  # JSON needs string keys
        'analysis': analysis
    }
    
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nResults exported to: {filepath}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark read_tree parallel scaling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python benchmark_parallel.py data.root
    python benchmark_parallel.py data.root --max-workers 16
    python benchmark_parallel.py data.root --timeout 30 --repeats 5
    python benchmark_parallel.py data.root --json results.json
    
Note: Timeout detection requires POSIX (Linux/macOS).
        """
    )
    parser.add_argument('filepath', help='Path to ROOT file')
    parser.add_argument('--treename', default='tree', help='Tree name (default: tree)')
    parser.add_argument('--max-workers', type=int, default=8,
                        help='Maximum workers to test (default: 8)')
    parser.add_argument('--workers', type=str, default=None,
                        help='Specific worker counts, comma-separated (e.g., "1,2,4,8")')
    parser.add_argument('--timeout', type=int, default=DEFAULT_TIMEOUT,
                        help=f'Timeout per read in seconds (default: {DEFAULT_TIMEOUT})')
    parser.add_argument('--repeats', type=int, default=DEFAULT_REPEATS,
                        help=f'Repetitions per worker count (default: {DEFAULT_REPEATS})')
    parser.add_argument('--json', type=str, metavar='FILE',
                        help='Export results to JSON file')
    parser.add_argument('--quiet', action='store_true',
                        help='Minimal output')
    
    args = parser.parse_args()
    
    # Validate file exists
    if not os.path.exists(args.filepath):
        print(f"Error: File not found: {args.filepath}")
        return 1
    
    # Determine worker counts to test
    if args.workers:
        worker_counts = [int(w.strip()) for w in args.workers.split(',')]
    else:
        worker_counts = [w for w in DEFAULT_WORKER_COUNTS if w <= args.max_workers]
    
    # Run benchmarks
    results = benchmark_workers(
        filepath=args.filepath,
        treename=args.treename,
        worker_counts=worker_counts,
        timeout=args.timeout,
        repeats=args.repeats,
        verbose=not args.quiet
    )
    
    # Analyze results
    analysis = analyze_results(results, verbose=not args.quiet)
    
    # Print summary
    if not args.quiet:
        print_summary(results, analysis)
    else:
        if analysis['optimal']:
            print(f"Optimal: num_workers={analysis['optimal']['num_workers']} "
                  f"({analysis['optimal']['mean_s']:.2f}s)")
        print(f"Issues: {len(analysis['issues'])}")
    
    # Export to JSON if requested
    if args.json:
        export_json(results, analysis, args.json, args.filepath, args.treename)
    
    # Always return 0 (report issues, don't fail)
    return 0


if __name__ == '__main__':
    sys.exit(main())
