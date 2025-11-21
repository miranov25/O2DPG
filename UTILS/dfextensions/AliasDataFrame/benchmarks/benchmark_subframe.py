#!/usr/bin/env python3
"""
benchmark_subframe.py - Benchmark and validation for AliasDataFrame subframes

Usage:
    python benchmark_subframe.py <input_file.root> [--treename tree] [--entries N]

Example:
    python benchmark_subframe.py dfcltrack_small.root --entries 100000

Tests:
    1. File loading performance
    2. Subframe detection and loading
    3. Invariant alias validation (T.mP3 == mP3_c, etc.)
    4. Alias materialization speed
    5. Missing key statistics
"""

import argparse
import sys
import os
import time
import warnings
import numpy as np

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dfextensions.AliasDataFrame import AliasDataFrame


def format_time(seconds):
    """Format time in human-readable form"""
    if seconds < 0.001:
        return f"{seconds*1e6:.1f} µs"
    elif seconds < 1:
        return f"{seconds*1000:.1f} ms"
    else:
        return f"{seconds:.2f} s"


def format_size(nbytes):
    """Format bytes in human-readable form"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


class BenchmarkResult:
    """Store and display benchmark results"""
    def __init__(self):
        self.results = []
        self.errors = []
        
    def add(self, name, passed, message="", time_sec=None):
        self.results.append({
            'name': name,
            'passed': passed,
            'message': message,
            'time': time_sec
        })
        if not passed:
            self.errors.append(name)
    
    def summary(self):
        passed = sum(1 for r in self.results if r['passed'])
        failed = len(self.results) - passed
        return passed, failed


def run_benchmark(filepath, treename="tree", max_entries=None, num_workers=4):
    """Run full benchmark suite"""
    
    results = BenchmarkResult()
    
    print("=" * 60)
    print(f"AliasDataFrame Subframe Benchmark")
    print("=" * 60)
    print(f"File: {filepath}")
    print(f"Tree: {treename}")
    print(f"Max entries: {max_entries or 'all'}")
    print()
    
    # =========================================================================
    # 1. File Loading
    # =========================================================================
    print("--- 1. File Loading ---")
    
    t0 = time.perf_counter()
    try:
        adf = AliasDataFrame.read_tree(
            filepath, 
            treename, 
            entry_stop=max_entries,
            num_workers=num_workers
        )
        t_load = time.perf_counter() - t0
        
        n_rows = len(adf.df)
        n_cols = len(adf.df.columns)
        mem_mb = adf.df.memory_usage(deep=True).sum() / 1024 / 1024
        
        print(f"  ✓ Loaded: {n_rows:,} rows, {n_cols} columns")
        print(f"  ✓ Memory: {mem_mb:.1f} MB")
        print(f"  ✓ Time: {format_time(t_load)}")
        print(f"  ✓ Speed: {n_rows / t_load:,.0f} rows/sec")
        
        results.add("File loading", True, f"{n_rows:,} rows in {format_time(t_load)}", t_load)
        
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        results.add("File loading", False, str(e))
        return results
    
    # =========================================================================
    # 2. Subframe Detection
    # =========================================================================
    print("\n--- 2. Subframe Detection ---")
    
    subframes = list(adf._subframes.subframes.keys())
    print(f"  Found {len(subframes)} subframes: {subframes}")
    
    for sf_name in subframes:
        sf = adf.get_subframe(sf_name)
        if sf is not None:
            entry = adf._subframes.get_entry(sf_name)
            idx_cols = entry['index']
            print(f"  ✓ {sf_name}: {len(sf.df):,} rows, index={idx_cols}")
            print(f"    Columns: {list(sf.df.columns)[:10]}{'...' if len(sf.df.columns) > 10 else ''}")
            results.add(f"Subframe {sf_name}", True, f"{len(sf.df):,} rows")
        else:
            print(f"  ✗ {sf_name}: Failed to load")
            results.add(f"Subframe {sf_name}", False, "Failed to load")
    
    # =========================================================================
    # 3. Join Correctness Validation
    # =========================================================================
    print("\n--- 3. Join Correctness Validation ---")
    
    sf_T = adf.get_subframe('T')
    if sf_T is not None:
        entry = adf._subframes.get_entry('T')
        idx_col = entry['index']
        if isinstance(idx_col, list):
            idx_col = idx_col[0]
        
        # Test lookups
        test_lookups = [
            ("T.mP3", "mP3"),
            ("T.mX", "mX"),
            ("T.mP4", "mP4"),
            ("T.dEdxTPC","dEdxTPC")
        ]
        
        for expr, col_name in test_lookups:
            if col_name not in sf_T.df.columns:
                continue
                
            alias_name = f"__test_{col_name}"
            adf.add_alias(alias_name, expr, dtype=np.float32)
            
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                adf.materialize_alias(alias_name)
            
            alias_vals = adf.df[alias_name].values
            n_valid = (~np.isnan(alias_vals)).sum()
            n_total = len(alias_vals)
            pct_valid = n_valid / n_total * 100
            
            print(f"  {expr}: {n_valid:,}/{n_total:,} valid ({pct_valid:.1f}%)")
            results.add(f"Lookup {col_name}", pct_valid > 90, f"{pct_valid:.1f}% valid")
        
        # =================================================================
        # TRUE INVARIANT TEST: Same key must give same value
        # =================================================================
        print("\n  Invariant test (same key → same value):")
        
        for expr, col_name in test_lookups:  # Test one column
            alias_name = f"__test_{col_name}"
            if alias_name not in adf.df.columns:
                continue
            
            # Group by index key, check std within each group
            # If join is correct, std should be 0 (same track → same value)
            grouped = adf.df.groupby(idx_col)[alias_name]
            
            # Std within each group (should be 0 if correct)
            group_stds = grouped.std()
            max_std = group_stds.max()
            mean_std = group_stds.mean()
            n_nonzero = (group_stds > 1e-6).sum()
            
            passed = max_std < 1e-5 or np.isnan(max_std)
            
            if passed:
                print(f"    ✓ {expr}: max_std_within_group={max_std:.2e} (CORRECT)")
            else:
                print(f"    ✗ {expr}: max_std_within_group={max_std:.2e}, groups_with_variance={n_nonzero}")
            
            results.add(f"Invariant {col_name}", passed, f"max_std={max_std:.2e}")
    else:
        print("  (Skipped - subframe T not found)")
    
    # =========================================================================
    # 4. Alias Materialization Speed
    # =========================================================================
    print("\n--- 4. Alias Materialization Speed ---")
    
    if sf_T is not None:
        # Test various alias patterns
        test_aliases = [
            ("simple_lookup", "T.mX"),
            ("expression", "mX - T.mX" if 'mX' in adf.df.columns else None),
        ]
        
        for alias_name, expr in test_aliases:
            if expr is None:
                continue
                
            # Check if columns exist
            try:
                adf.add_alias(f"__bench_{alias_name}", expr, dtype=np.float32)
                
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    t0 = time.perf_counter()
                    adf.materialize_alias(f"__bench_{alias_name}")
                    t_mat = time.perf_counter() - t0
                
                n_missing = 0
                for warn in w:
                    if 'not found' in str(warn.message):
                        # Extract count from warning
                        msg = str(warn.message)
                        try:
                            n_missing = int(msg.split()[2].replace(',', ''))
                        except:
                            pass
                
                n_valid = len(adf.df) - adf.df[f"__bench_{alias_name}"].isna().sum()
                
                print(f"  ✓ {alias_name}: {format_time(t_mat)}, valid={n_valid:,}/{len(adf.df):,}")
                results.add(f"Alias {alias_name}", True, format_time(t_mat), t_mat)
                
            except Exception as e:
                print(f"  ✗ {alias_name}: {e}")
                results.add(f"Alias {alias_name}", False, str(e))
    
    # =========================================================================
    # 5. Missing Key Statistics
    # =========================================================================
    print("\n--- 5. Missing Key Statistics ---")
    
    for sf_name in subframes:
        entry = adf._subframes.get_entry(sf_name)
        if entry:
            idx_cols = entry['index']
            if isinstance(idx_cols, str):
                idx_cols = [idx_cols]
            
            # Check how many main frame keys exist in subframe
            sf = entry['frame']
            
            if all(col in adf.df.columns for col in idx_cols):
                if len(idx_cols) == 1:
                    main_keys = set(adf.df[idx_cols[0]].unique())
                    sub_keys = set(sf.df[idx_cols[0]].unique())
                else:
                    main_keys = set(map(tuple, adf.df[idx_cols].values))
                    sub_keys = set(map(tuple, sf.df[idx_cols].values))
                
                n_in_both = len(main_keys & sub_keys)
                n_main_only = len(main_keys - sub_keys)
                n_sub_only = len(sub_keys - main_keys)
                
                coverage = n_in_both / len(main_keys) * 100 if main_keys else 0
                
                print(f"  {sf_name}:")
                print(f"    Main keys: {len(main_keys):,}")
                print(f"    Subframe keys: {len(sub_keys):,}")
                print(f"    Coverage: {coverage:.1f}% ({n_in_both:,} matched)")
                if n_main_only > 0:
                    print(f"    Missing in subframe: {n_main_only:,}")
                
                results.add(f"Coverage {sf_name}", coverage > 50, f"{coverage:.1f}%")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    passed, failed = results.summary()
    print(f"SUMMARY: {passed} passed, {failed} failed")
    
    if failed > 0:
        print(f"Failed tests: {results.errors}")
    
    print("=" * 60)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark AliasDataFrame subframes")
    parser.add_argument("filepath", help="Path to ROOT file")
    parser.add_argument("--treename", default="tree", help="Tree name (default: tree)")
    parser.add_argument("--entries", type=int, default=None, help="Max entries to load")
    parser.add_argument("--workers", type=int, default=4, help="Number of workers")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.filepath):
        print(f"Error: File not found: {args.filepath}")
        sys.exit(1)
    
    results = run_benchmark(
        args.filepath,
        treename=args.treename,
        max_entries=args.entries,
        num_workers=args.workers
    )
    
    passed, failed = results.summary()
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
