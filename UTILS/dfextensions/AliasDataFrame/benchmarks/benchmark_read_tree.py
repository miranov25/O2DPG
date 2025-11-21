#!/usr/bin/env python3
"""
AliasDataFrame Read Performance Benchmark

Usage: 
    python benchmark_read_tree.py <filename> [treename] [entry_stop]
    python benchmark_read_tree.py <filename> [treename] --entry-start=N --entry-stop=M

Examples:
    python benchmark_read_tree.py data.root tree 1000000
    python benchmark_read_tree.py data.root tree  # full file
    python benchmark_read_tree.py data.root tree --entry-start=0 --entry-stop=100000
"""

import sys
import os
import time
import gc
import tracemalloc
import argparse
import warnings
import pandas as pd
import numpy as np

# Suppress expected float16 overflow warnings (compression precision)
warnings.filterwarnings('ignore', message='overflow encountered in cast')

# Add parent directory to path for AliasDataFrame import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_peak_memory_mb(func, *args, **kwargs):
    """Run function and return (result, peak_memory_mb, duration_s)"""
    gc.collect()
    tracemalloc.start()
    
    t0 = time.time()
    result = func(*args, **kwargs)
    duration = time.time() - t0
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return result, peak / 1024 / 1024, duration


def test1_uproot_oneshot(filename, treename, entry_start, entry_stop):
    """Test 1: Pure uproot one-shot arrays()"""
    import uproot
    
    with uproot.open(filename) as f:
        tree = f[treename]
        arrays = tree.arrays(
            library="np",
            entry_start=entry_start,
            entry_stop=entry_stop
        )
    df = pd.DataFrame(arrays)
    return df


def test2_uproot_branch_by_branch(filename, treename, entry_start, entry_stop):
    """Test 2: Branch-by-branch read (no conversion)"""
    import uproot
    
    with uproot.open(filename) as f:
        tree = f[treename]
        arrays = {}
        for branch_name in tree.keys():
            arrays[branch_name] = tree[branch_name].array(
                library="np",
                entry_start=entry_start,
                entry_stop=entry_stop
            )
    df = pd.DataFrame(arrays)
    return df


def test3_uproot_branch_with_conversion(filename, treename, entry_start, entry_stop, dtype_hints=None):
    """Test 3: Branch-by-branch read with dtype conversion"""
    import uproot
    
    if dtype_hints is None:
        dtype_hints = {}
    
    with uproot.open(filename) as f:
        tree = f[treename]
        arrays = {}
        for branch_name in tree.keys():
            arr = tree[branch_name].array(
                library="np",
                entry_start=entry_start,
                entry_stop=entry_stop
            )
            if branch_name in dtype_hints:
                arr = arr.astype(dtype_hints[branch_name])
            arrays[branch_name] = arr
    df = pd.DataFrame(arrays)
    return df


def test4_aliasdf_read_tree(filename, treename, entry_start, entry_stop):
    """Test 4: AliasDataFrame.read_tree (Phase 2 optimized with entry_range)"""
    from ..AliasDataFrame import AliasDataFrame
    
    # Phase 2: Now supports entry_start/entry_stop
    adf = AliasDataFrame.read_tree(
        filename, 
        treename,
        entry_start=entry_start,
        entry_stop=entry_stop,
        num_workers=8
    )
    return adf.df


def test5_uproot_oneshot_threaded(filename, treename, entry_start, entry_stop, num_workers=8):
    """Test 5: Pure uproot one-shot with ThreadPoolExecutor"""
    import uproot
    import concurrent.futures
    
    with uproot.open(filename) as f:
        tree = f[treename]
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            arrays = tree.arrays(
                library="np",
                entry_start=entry_start,
                entry_stop=entry_stop,
                decompression_executor=executor,
                interpretation_executor=executor
            )
    df = pd.DataFrame(arrays)
    return df


def test7_uproot_branch_threaded(filename, treename, entry_start, entry_stop, dtype_hints=None, num_workers=8):
    """Test 7: Branch-by-branch with ThreadPoolExecutor per branch"""
    import uproot
    import concurrent.futures
    
    if dtype_hints is None:
        dtype_hints = {}
    
    with uproot.open(filename) as f:
        tree = f[treename]
        branch_names = list(tree.keys())
        
        def read_branch(branch_name):
            arr = tree[branch_name].array(
                library="np",
                entry_start=entry_start,
                entry_stop=entry_stop
            )
            if branch_name in dtype_hints:
                arr = arr.astype(dtype_hints[branch_name])
            return branch_name, arr
        
        # Read branches in parallel
        arrays = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(read_branch, name): name for name in branch_names}
            for future in concurrent.futures.as_completed(futures):
                name, arr = future.result()
                arrays[name] = arr
    
    df = pd.DataFrame(arrays)
    return df


def test8_uproot_branch_threaded_convert(filename, treename, entry_start, entry_stop, dtype_hints, num_workers=8):
    """Test 8: Branch-by-branch threaded WITH dtype conversion"""
    return test7_uproot_branch_threaded(filename, treename, entry_start, entry_stop, dtype_hints, num_workers)


def test6_uproot_pandas_direct(filename, treename, entry_start, entry_stop):
    """Test 6: Pure uproot with library='pd' (original method)"""
    import uproot
    
    with uproot.open(filename) as f:
        tree = f[treename]
        df = tree.arrays(
            library="pd",
            entry_start=entry_start,
            entry_stop=entry_stop
        )
    return df


def run_benchmark(filename, treename="tree", entry_start=None, entry_stop=None, num_workers=8):
    """Run all benchmarks and report results"""
    
    print("=" * 70)
    print("AliasDataFrame Read Performance Benchmark")
    print("=" * 70)
    print(f"File: {filename}")
    print(f"Tree: {treename}")
    print(f"Range: {entry_start or 0} : {entry_stop or 'end'}")
    print(f"Workers: {num_workers}")
    print("=" * 70)
    
    # Get file info
    import uproot
    with uproot.open(filename) as f:
        tree = f[treename]
        total_entries = tree.num_entries
        n_branches = len(tree.keys())
        branch_names = list(tree.keys())
        
        # Determine actual entries to read
        start = entry_start or 0
        stop = entry_stop or total_entries
        entries_to_read = stop - start
        
    print(f"Total entries in tree: {total_entries:,}")
    print(f"Entries to read: {entries_to_read:,}")
    print(f"Branches: {n_branches}")
    print("=" * 70)
    
    results = []
    
    # Test 6: Original pandas direct (baseline comparison)
    print("\n[6] Uproot arrays(library='pd') - original method")
    df, peak_mem, duration = get_peak_memory_mb(
        test6_uproot_pandas_direct, filename, treename, entry_start, entry_stop
    )
    df_mem = df.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"    Time: {duration:.1f}s")
    print(f"    Peak memory: {peak_mem:.0f} MB")
    print(f"    DataFrame: {df_mem:.0f} MB")
    results.append(("6-pd-direct", duration, peak_mem, df_mem))
    del df
    gc.collect()
    
    # Test 1: One-shot numpy
    print("\n[1] Uproot one-shot arrays(library='np') -> DataFrame")
    df, peak_mem, duration = get_peak_memory_mb(
        test1_uproot_oneshot, filename, treename, entry_start, entry_stop
    )
    df_mem = df.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"    Time: {duration:.1f}s")
    print(f"    Peak memory: {peak_mem:.0f} MB")
    print(f"    DataFrame: {df_mem:.0f} MB")
    results.append(("1-np-oneshot", duration, peak_mem, df_mem))
    del df
    gc.collect()
    
    # Test 2: Branch-by-branch
    print("\n[2] Uproot branch-by-branch (no conversion)")
    df, peak_mem, duration = get_peak_memory_mb(
        test2_uproot_branch_by_branch, filename, treename, entry_start, entry_stop
    )
    df_mem = df.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"    Time: {duration:.1f}s")
    print(f"    Peak memory: {peak_mem:.0f} MB")
    print(f"    DataFrame: {df_mem:.0f} MB")
    results.append(("2-branch", duration, peak_mem, df_mem))
    del df
    gc.collect()
    
    # Test 3: Branch-by-branch with conversion
    # Create dtype_hints: convert every other float column to float16
    dtype_hints = {}
    with uproot.open(filename) as f:
        tree = f[treename]
        for i, branch_name in enumerate(tree.keys()):
            if i % 2 == 0:
                dtype_hints[branch_name] = np.float16
    
    print(f"\n[3] Uproot branch-by-branch with dtype conversion")
    print(f"    (converting {len(dtype_hints)} columns to float16)")
    df, peak_mem, duration = get_peak_memory_mb(
        test3_uproot_branch_with_conversion, filename, treename, 
        entry_start, entry_stop, dtype_hints
    )
    df_mem = df.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"    Time: {duration:.1f}s")
    print(f"    Peak memory: {peak_mem:.0f} MB")
    print(f"    DataFrame: {df_mem:.0f} MB")
    results.append(("3-convert", duration, peak_mem, df_mem))
    del df
    gc.collect()
    
    # Test 5: One-shot with threads
    print(f"\n[5] Uproot one-shot with ThreadPoolExecutor ({num_workers} workers)")
    df, peak_mem, duration = get_peak_memory_mb(
        test5_uproot_oneshot_threaded, filename, treename, entry_start, entry_stop, num_workers
    )
    df_mem = df.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"    Time: {duration:.1f}s")
    print(f"    Peak memory: {peak_mem:.0f} MB")
    print(f"    DataFrame: {df_mem:.0f} MB")
    results.append(("5-threaded", duration, peak_mem, df_mem))
    del df
    gc.collect()
    
    # Test 7: Branch-by-branch with threads (no conversion)
    print(f"\n[7] Branch-by-branch with ThreadPoolExecutor ({num_workers} workers)")
    df, peak_mem, duration = get_peak_memory_mb(
        test7_uproot_branch_threaded, filename, treename, entry_start, entry_stop, None, num_workers
    )
    df_mem = df.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"    Time: {duration:.1f}s")
    print(f"    Peak memory: {peak_mem:.0f} MB")
    print(f"    DataFrame: {df_mem:.0f} MB")
    results.append(("7-branch-thr", duration, peak_mem, df_mem))
    del df
    gc.collect()
    
    # Test 8: Branch-by-branch threaded with conversion
    print(f"\n[8] Branch-by-branch threaded with dtype conversion ({num_workers} workers)")
    print(f"    (converting {len(dtype_hints)} columns to float16)")
    df, peak_mem, duration = get_peak_memory_mb(
        test8_uproot_branch_threaded_convert, filename, treename, entry_start, entry_stop, dtype_hints, num_workers
    )
    df_mem = df.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"    Time: {duration:.1f}s")
    print(f"    Peak memory: {peak_mem:.0f} MB")
    print(f"    DataFrame: {df_mem:.0f} MB")
    results.append(("8-branch-thr-conv", duration, peak_mem, df_mem))
    del df
    gc.collect()
    
    # Test 4: AliasDataFrame.read_tree (Phase 2 - now supports entry_range!)
    print("\n[4] AliasDataFrame.read_tree (optimized, num_workers=8)")
    try:
        df, peak_mem, duration = get_peak_memory_mb(
            test4_aliasdf_read_tree, filename, treename, entry_start, entry_stop
        )
        df_mem = df.memory_usage(deep=True).sum() / 1024 / 1024
        print(f"    Time: {duration:.1f}s")
        print(f"    Peak memory: {peak_mem:.0f} MB")
        print(f"    DataFrame: {df_mem:.0f} MB")
        results.append(("4-aliasdf", duration, peak_mem, df_mem))
        del df
        gc.collect()
    except Exception as e:
        print(f"    ERROR: {e}")
        results.append(("4-aliasdf", None, None, None))
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Test':<15} {'Time (s)':<12} {'Peak (MB)':<12} {'Final (MB)':<12}")
    print("-" * 51)
    for name, t, peak, final in results:
        if t is not None:
            print(f"{name:<15} {t:<12.1f} {peak:<12.0f} {final:<12.0f}")
        else:
            print(f"{name:<15} {'SKIPPED':<12} {'-':<12} {'-':<12}")
    
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    # Find results by name
    def get_result(name_prefix):
        for name, t, peak, final in results:
            if name.startswith(name_prefix):
                return t, peak, final
        return None, None, None
    
    t6, _, _ = get_result("6-pd")    # pandas direct
    t1, _, _ = get_result("1-np")    # numpy oneshot
    t2, p2, _ = get_result("2-branch") # branch-by-branch
    t3, p3, f3 = get_result("3-conv") # with conversion
    t5, p5, _ = get_result("5-thread") # threaded oneshot
    t7, p7, _ = get_result("7-branch-thr") # threaded branch
    t8, p8, f8 = get_result("8-branch-thr-conv") # threaded branch + convert
    t4, _, _ = get_result("4-alias")  # aliasdf
    
    if t6 and t1:
        improvement = ((t6 - t1) / t6) * 100
        print(f"numpy vs pandas direct: {improvement:+.0f}% ({'faster' if improvement > 0 else 'slower'})")
    
    if t1 and t2:
        overhead = ((t2 - t1) / t1) * 100
        print(f"Branch-by-branch overhead: {overhead:+.0f}% vs one-shot")
    
    if t1 and t3:
        overhead = ((t3 - t1) / t1) * 100
        print(f"With dtype conversion: {overhead:+.0f}% vs one-shot")
    
    if t1 and t5:
        benefit = ((t1 - t5) / t1) * 100
        print(f"Threaded one-shot: {benefit:+.0f}% vs one-shot")
    
    if t1 and t7:
        benefit = ((t1 - t7) / t1) * 100
        print(f"Threaded branch-by-branch: {benefit:+.0f}% vs one-shot")
    
    if t1 and t8:
        benefit = ((t1 - t8) / t1) * 100
        print(f"Threaded branch + convert: {benefit:+.0f}% vs one-shot")
    
    if t1 and t4:
        overhead = ((t4 - t1) / t1) * 100
        print(f"AliasDataFrame overhead: {overhead:+.0f}% vs numpy one-shot")
    
    if t6 and t4:
        overhead = ((t4 - t6) / t6) * 100
        print(f"AliasDataFrame overhead: {overhead:+.0f}% vs pandas direct")
    
    # Memory analysis
    _, p1, f1 = get_result("1-np")
    if p1 and f3:
        mem_saving = ((f1 - f3) / f1) * 100
        print(f"\nMemory saving with float16 conversion: {mem_saving:.0f}%")
    
    if p1 and p7:
        peak_saving = ((p1 - p7) / p1) * 100
        print(f"Peak memory saving (threaded branch): {peak_saving:.0f}%")
    
    if p5 and p8:
        peak_saving = ((p5 - p8) / p5) * 100
        print(f"Peak memory saving (thr-branch vs thr-oneshot): {peak_saving:.0f}%")
    
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    
    # Determine best approach based on results
    best_time = None
    best_name = None
    best_peak = None
    
    # Find best time with acceptable memory
    for name, t, peak, final in results:
        if t is None:
            continue
        if best_time is None or t < best_time:
            best_time = t
            best_name = name
            best_peak = peak
    
    print(f"Fastest approach: {best_name} ({best_time:.1f}s)")
    
    # Check threaded branch-by-branch specifically
    if t7 and t8 and p7 and p8:
        if t7 < t1 * 0.5 and p7 < p1 * 0.5:
            print("\n✓✓ THREADED BRANCH-BY-BRANCH is the winner!")
            print(f"   Speed: {t7:.1f}s (vs {t1:.1f}s one-shot)")
            print(f"   Peak memory: {p7:.0f} MB (vs {p1:.0f} MB one-shot)")
            print("   → Use this as default with num_workers>1")
        elif t7 < t1:
            print("\n✓ Threaded branch-by-branch provides speedup with low memory")
            print("   → Consider as default when num_workers>1")
        else:
            print("\n✗ Threaded branch-by-branch slower than expected")
            print("   → May be I/O bound or GIL contention")
    
    if t2 and t1 and t2 < t1 * 1.5:
        print("\n✓ Non-threaded branch-by-branch is acceptable (<50% overhead)")
        print("  → Use as fallback when num_workers=1")
    
    if t5 and t1 and t5 < t1 * 0.5:
        print(f"\n✓ Threaded one-shot is very fast but high memory ({p5:.0f} MB)")
        print("  → Only for systems with large RAM")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark AliasDataFrame read performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s data.root tree 1000000
    %(prog)s data.root tree --entry-stop=100000
    %(prog)s data.root tree --entry-start=1000000 --entry-stop=2000000
        """
    )
    parser.add_argument("filename", help="ROOT file path")
    parser.add_argument("treename", nargs="?", default="tree", help="Tree name (default: tree)")
    parser.add_argument("entry_stop_pos", nargs="?", type=int, default=None,
                        help="Stop at this entry (positional, for convenience)")
    parser.add_argument("--entry-start", type=int, default=None,
                        help="Start from this entry")
    parser.add_argument("--entry-stop", type=int, default=None,
                        help="Stop at this entry")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of worker threads (default: 8)")
    
    args = parser.parse_args()
    
    # Handle positional entry_stop for convenience
    entry_stop = args.entry_stop or args.entry_stop_pos
    
    run_benchmark(
        args.filename,
        args.treename,
        args.entry_start,
        entry_stop,
        args.workers
    )


if __name__ == "__main__":
    main()
