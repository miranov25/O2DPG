#!/usr/bin/env python3
"""
diagnose_read_performance.py - Diagnose AliasDataFrame read slowdowns

Purpose: Identify root cause of intermittent read performance issues.

Usage:
    python diagnose_read_performance.py data.root
    python diagnose_read_performance.py data.root --test workers
    python diagnose_read_performance.py data.root --test all --json results.json

Tests:
    workers  - Thread scaling (1,2,4,8,16 workers)
    cache    - Cold vs warm cache (5 sequential reads)
    local    - Network vs local storage comparison
    overhead - Raw uproot vs AliasDataFrame overhead
    all      - Run all tests (default)

Output:
    Console summary with diagnosis and recommendations.
    Optional JSON export for analysis.
"""

import argparse
import gc
import json
import os
import platform
import shutil
import sys
import tempfile
import time
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_system_info():
    """Collect system information for diagnostics."""
    info = {
        'hostname': platform.node(),
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': os.cpu_count(),
        'timestamp': datetime.now().isoformat(),
    }
    
    # Check if running on network filesystem
    try:
        import subprocess
        result = subprocess.run(['df', '-T', '.'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                parts = lines[1].split()
                if len(parts) >= 2:
                    info['filesystem_type'] = parts[1]
    except:
        pass
    
    return info


def format_time(seconds):
    """Format time consistently."""
    if seconds < 0.1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 10:
        return f"{seconds:.2f}s"
    else:
        return f"{seconds:.1f}s"


def test_workers(filepath, treename, workers_list=None, entry_stop=None):
    """
    Test performance with different worker counts.
    
    Identifies: Thread contention, GIL issues, optimal worker count.
    """
    from AliasDataFrame import AliasDataFrame
    
    if workers_list is None:
        workers_list = [1, 2, 4, 8, 16]
    
    print("\n" + "=" * 60)
    print("TEST: WORKER SCALING")
    print("=" * 60)
    print("Purpose: Find optimal worker count, detect thread contention")
    print()
    
    results = []
    baseline_time = None
    
    for n_workers in workers_list:
        gc.collect()
        
        t0 = time.time()
        try:
            adf = AliasDataFrame.read_tree(
                filepath, treename, 
                num_workers=n_workers,
                entry_stop=entry_stop
            )
            elapsed = time.time() - t0
            n_rows = len(adf.df)
            
            if baseline_time is None:
                baseline_time = elapsed
                speedup = 1.0
            else:
                speedup = baseline_time / elapsed
            
            result = {
                'workers': n_workers,
                'time_s': elapsed,
                'rows': n_rows,
                'speedup': speedup,
                'rows_per_sec': n_rows / elapsed,
            }
            results.append(result)
            
            speedup_str = f"{speedup:.2f}x" if speedup != 1.0 else "(baseline)"
            print(f"  workers={n_workers:2d}: {format_time(elapsed):>8s}  "
                  f"speedup={speedup_str:>8s}  "
                  f"({n_rows:,} rows, {result['rows_per_sec']:,.0f} rows/s)")
            
            del adf
            
        except Exception as e:
            print(f"  workers={n_workers:2d}: ERROR - {e}")
            results.append({
                'workers': n_workers,
                'error': str(e),
            })
    
    # Analysis
    print()
    successful = [r for r in results if 'time_s' in r]
    if successful:
        best = min(successful, key=lambda x: x['time_s'])
        worst = max(successful, key=lambda x: x['time_s'])
        
        print("Analysis:")
        print(f"  Best:  workers={best['workers']} ({format_time(best['time_s'])})")
        print(f"  Worst: workers={worst['workers']} ({format_time(worst['time_s'])})")
        
        # Check for negative scaling (more workers = slower)
        if len(successful) >= 2:
            last = successful[-1]
            first = successful[0]
            if last['time_s'] > first['time_s'] * 1.5:
                print()
                print("⚠️  WARNING: Negative scaling detected!")
                print("   More workers = slower. Possible causes:")
                print("   - Thread contention")
                print("   - I/O bottleneck (network/disk)")
                print("   - Shared server load")
    
    return {'test': 'workers', 'results': results}


def test_cache(filepath, treename, n_reads=5, n_workers=4, entry_stop=None):
    """
    Test cold vs warm cache performance.
    
    Identifies: Filesystem caching effects, cold start overhead.
    """
    from AliasDataFrame import AliasDataFrame
    
    print("\n" + "=" * 60)
    print("TEST: COLD VS WARM CACHE")
    print("=" * 60)
    print(f"Purpose: Measure filesystem caching effect ({n_reads} sequential reads)")
    print()
    
    results = []
    
    for i in range(n_reads):
        gc.collect()
        
        t0 = time.time()
        adf = AliasDataFrame.read_tree(
            filepath, treename,
            num_workers=n_workers,
            entry_stop=entry_stop
        )
        elapsed = time.time() - t0
        n_rows = len(adf.df)
        
        label = "COLD" if i == 0 else f"WARM-{i}"
        
        result = {
            'read': i + 1,
            'label': label,
            'time_s': elapsed,
            'rows': n_rows,
        }
        results.append(result)
        
        print(f"  Read {i+1} ({label:6s}): {format_time(elapsed):>8s}  ({n_rows:,} rows)")
        
        del adf
    
    # Analysis
    print()
    if len(results) >= 2:
        cold = results[0]['time_s']
        warm_avg = sum(r['time_s'] for r in results[1:]) / (len(results) - 1)
        cache_benefit = (cold - warm_avg) / cold * 100
        
        print("Analysis:")
        print(f"  Cold read:     {format_time(cold)}")
        print(f"  Warm average:  {format_time(warm_avg)}")
        print(f"  Cache benefit: {cache_benefit:.0f}%")
        
        if cache_benefit > 50:
            print()
            print("✓ Significant caching benefit detected.")
            print("  First read is slower due to filesystem cache miss.")
            print("  Consider: Pre-warming cache, using local storage.")
        elif cold > warm_avg * 3:
            print()
            print("⚠️  WARNING: Extreme cold start penalty!")
            print("   First read is >3x slower. Possible network filesystem.")
    
    return {'test': 'cache', 'results': results}


def test_local_vs_network(filepath, treename, n_workers=4, entry_stop=None):
    """
    Compare network vs local storage performance.
    
    Identifies: Network I/O bottleneck.
    """
    from AliasDataFrame import AliasDataFrame
    
    print("\n" + "=" * 60)
    print("TEST: LOCAL VS NETWORK STORAGE")
    print("=" * 60)
    print("Purpose: Identify network I/O bottleneck")
    print()
    
    results = {}
    
    # Test original location
    print("Testing original location...")
    gc.collect()
    t0 = time.time()
    adf = AliasDataFrame.read_tree(
        filepath, treename,
        num_workers=n_workers,
        entry_stop=entry_stop
    )
    original_time = time.time() - t0
    n_rows = len(adf.df)
    del adf
    
    results['original'] = {
        'path': filepath,
        'time_s': original_time,
        'rows': n_rows,
    }
    print(f"  Original: {format_time(original_time):>8s}  ({filepath})")
    
    # Copy to local temp directory
    print("Copying to local storage...")
    local_path = os.path.join(tempfile.gettempdir(), f"diag_{os.getpid()}.root")
    
    t0 = time.time()
    shutil.copy(filepath, local_path)
    copy_time = time.time() - t0
    file_size = os.path.getsize(filepath)
    copy_speed = file_size / copy_time / 1e6  # MB/s
    
    print(f"  Copy time: {format_time(copy_time)} ({copy_speed:.0f} MB/s)")
    
    # Test local
    gc.collect()
    t0 = time.time()
    adf = AliasDataFrame.read_tree(
        local_path, treename,
        num_workers=n_workers,
        entry_stop=entry_stop
    )
    local_time = time.time() - t0
    del adf
    
    results['local'] = {
        'path': local_path,
        'time_s': local_time,
        'rows': n_rows,
    }
    print(f"  Local:    {format_time(local_time):>8s}  ({local_path})")
    
    # Cleanup
    try:
        os.remove(local_path)
    except:
        pass
    
    # Analysis
    print()
    speedup = original_time / local_time
    print("Analysis:")
    print(f"  Original: {format_time(original_time)}")
    print(f"  Local:    {format_time(local_time)}")
    print(f"  Speedup:  {speedup:.2f}x")
    
    results['speedup'] = speedup
    results['copy_time_s'] = copy_time
    results['copy_speed_mbs'] = copy_speed
    
    if speedup > 2:
        print()
        print("⚠️  WARNING: Network storage is >2x slower!")
        print("   Recommendation: Copy data to local /tmp before processing.")
        print(f"   Copy overhead: {format_time(copy_time)} (amortized over multiple reads)")
    elif speedup > 1.5:
        print()
        print("✓ Moderate network overhead detected.")
        print("  Consider local copy for repeated processing.")
    else:
        print()
        print("✓ Storage location has minimal impact.")
    
    return {'test': 'local_vs_network', 'results': results}


def test_overhead(filepath, treename, n_workers=4, entry_stop=None):
    """
    Compare raw uproot vs AliasDataFrame overhead.
    
    Identifies: AliasDataFrame-specific overhead.
    """
    import uproot
    from AliasDataFrame import AliasDataFrame
    
    print("\n" + "=" * 60)
    print("TEST: UPROOT VS ALIASDATAFRAME OVERHEAD")
    print("=" * 60)
    print("Purpose: Measure AliasDataFrame processing overhead")
    print()
    
    results = {}
    
    # Test raw uproot
    print("Testing raw uproot...")
    gc.collect()
    t0 = time.time()
    with uproot.open(filepath) as f:
        tree = f[treename]
        if entry_stop:
            df = tree.arrays(library="pd", entry_stop=entry_stop)
        else:
            df = tree.arrays(library="pd")
    uproot_time = time.time() - t0
    n_rows = len(df)
    del df
    
    results['uproot'] = {
        'time_s': uproot_time,
        'rows': n_rows,
    }
    print(f"  Raw uproot:      {format_time(uproot_time):>8s}  ({n_rows:,} rows)")
    
    # Test AliasDataFrame
    print("Testing AliasDataFrame...")
    gc.collect()
    t0 = time.time()
    adf = AliasDataFrame.read_tree(
        filepath, treename,
        num_workers=n_workers,
        entry_stop=entry_stop
    )
    adf_time = time.time() - t0
    del adf
    
    results['aliasdataframe'] = {
        'time_s': adf_time,
        'rows': n_rows,
        'num_workers': n_workers,
    }
    print(f"  AliasDataFrame:  {format_time(adf_time):>8s}  (workers={n_workers})")
    
    # Analysis
    print()
    overhead = (adf_time - uproot_time) / uproot_time * 100
    ratio = adf_time / uproot_time
    
    print("Analysis:")
    print(f"  Raw uproot:     {format_time(uproot_time)}")
    print(f"  AliasDataFrame: {format_time(adf_time)}")
    print(f"  Overhead:       {overhead:+.0f}% ({ratio:.2f}x)")
    
    results['overhead_pct'] = overhead
    results['ratio'] = ratio
    
    if ratio < 0.8:
        print()
        print("✓ AliasDataFrame is FASTER than raw uproot!")
        print("  Threaded branch-by-branch reading provides speedup.")
    elif ratio < 1.2:
        print()
        print("✓ AliasDataFrame overhead is minimal (<20%).")
    elif ratio < 2.0:
        print()
        print("⚠️  Moderate overhead (20-100%).")
        print("   Consider: Adjusting num_workers, checking subframe loading.")
    else:
        print()
        print("⚠️  WARNING: High overhead (>100%)!")
        print("   Possible causes:")
        print("   - Thread contention (try fewer workers)")
        print("   - Subframe loading overhead")
        print("   - Schema processing")
    
    return {'test': 'overhead', 'results': results}


def print_recommendations(all_results):
    """Print overall recommendations based on all test results."""
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    recommendations = []
    
    # Check worker scaling
    if 'workers' in all_results:
        worker_results = all_results['workers'].get('results', [])
        successful = [r for r in worker_results if 'time_s' in r]
        if successful:
            best = min(successful, key=lambda x: x['time_s'])
            recommendations.append(f"• Use num_workers={best['workers']} for best performance")
            
            # Check for negative scaling
            if len(successful) >= 2:
                first = successful[0]
                last = successful[-1]
                if last['time_s'] > first['time_s'] * 1.5:
                    recommendations.append("• Avoid high worker counts - negative scaling detected")
    
    # Check cache effect
    if 'cache' in all_results:
        cache_results = all_results['cache'].get('results', [])
        if len(cache_results) >= 2:
            cold = cache_results[0]['time_s']
            warm_avg = sum(r['time_s'] for r in cache_results[1:]) / (len(cache_results) - 1)
            if cold > warm_avg * 2:
                recommendations.append("• First read is slow due to cache miss - consider pre-warming")
    
    # Check local vs network
    if 'local_vs_network' in all_results:
        local_results = all_results['local_vs_network'].get('results', {})
        speedup = local_results.get('speedup', 1.0)
        if speedup > 2:
            recommendations.append("• Copy data to local /tmp before processing (>2x speedup)")
        elif speedup > 1.5:
            recommendations.append("• Consider local storage for repeated processing")
    
    # Check overhead
    if 'overhead' in all_results:
        overhead_results = all_results['overhead'].get('results', {})
        ratio = overhead_results.get('ratio', 1.0)
        if ratio > 2.0:
            recommendations.append("• High ADF overhead - check subframe loading, reduce workers")
    
    if recommendations:
        for rec in recommendations:
            print(rec)
    else:
        print("✓ No significant issues detected.")
        print("  Performance appears normal.")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose AliasDataFrame read performance issues",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python diagnose_read_performance.py data.root
    python diagnose_read_performance.py data.root --test workers
    python diagnose_read_performance.py data.root --test all --json results.json
    python diagnose_read_performance.py data.root --entries 100000  # Quick test
        """
    )
    parser.add_argument("filepath", help="Path to ROOT file")
    parser.add_argument("--treename", default="tree", help="Tree name (default: tree)")
    parser.add_argument("--test", choices=['all', 'workers', 'cache', 'local', 'overhead'],
                        default='all', help="Test to run (default: all)")
    parser.add_argument("--entries", type=int, default=None,
                        help="Limit entries for faster testing")
    parser.add_argument("--workers", type=str, default="1,2,4,8,16",
                        help="Worker counts to test (default: 1,2,4,8,16)")
    parser.add_argument("--json", type=str, default=None,
                        help="Export results to JSON file")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.filepath):
        print(f"Error: File not found: {args.filepath}")
        sys.exit(1)
    
    # Parse worker list
    workers_list = [int(w.strip()) for w in args.workers.split(',')]
    
    # Header
    print("=" * 60)
    print("ALIASDATAFRAME READ PERFORMANCE DIAGNOSTIC")
    print("=" * 60)
    
    file_size = os.path.getsize(args.filepath)
    print(f"File:      {args.filepath}")
    print(f"Size:      {file_size / 1e6:.1f} MB")
    print(f"Tree:      {args.treename}")
    if args.entries:
        print(f"Entries:   {args.entries:,} (limited)")
    print(f"Tests:     {args.test}")
    print()
    
    system_info = get_system_info()
    print(f"Host:      {system_info['hostname']}")
    print(f"Platform:  {system_info['platform']}")
    print(f"Python:    {system_info['python_version']}")
    print(f"CPUs:      {system_info['cpu_count']}")
    if 'filesystem_type' in system_info:
        print(f"FS Type:   {system_info['filesystem_type']}")
    
    # Run tests
    all_results = {
        'system_info': system_info,
        'file': args.filepath,
        'file_size_bytes': file_size,
        'treename': args.treename,
        'entry_stop': args.entries,
    }
    
    try:
        if args.test in ['all', 'workers']:
            result = test_workers(args.filepath, args.treename, 
                                  workers_list, args.entries)
            all_results['workers'] = result
        
        if args.test in ['all', 'cache']:
            result = test_cache(args.filepath, args.treename, 
                                n_workers=4, entry_stop=args.entries)
            all_results['cache'] = result
        
        if args.test in ['all', 'local']:
            result = test_local_vs_network(args.filepath, args.treename,
                                           n_workers=4, entry_stop=args.entries)
            all_results['local_vs_network'] = result
        
        if args.test in ['all', 'overhead']:
            result = test_overhead(args.filepath, args.treename,
                                   n_workers=4, entry_stop=args.entries)
            all_results['overhead'] = result
        
        # Print recommendations
        print_recommendations(all_results)
        
    except KeyboardInterrupt:
        print("\n\nDiagnostic interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during diagnostic: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Export JSON if requested
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"Results exported to: {args.json}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
