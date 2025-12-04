#!/usr/bin/env python3
"""
benchmark_rdf.py - Benchmark RDataFrame vs TTree::Draw vs AliasDataFrame

Compares performance of three approaches for evaluating aliases:
1. RDataFrame (JIT-compiled C++)
2. TTree::Draw (interpreted)
3. AliasDataFrame (pandas/numpy)

Usage:
    python benchmark_rdf.py synthetic_data.root
    python benchmark_rdf.py data.root --aliases dyC2 dzC2
    python benchmark_rdf.py data.root --mt --iterations 5
    python benchmark_rdf.py data.root --json results.json

Exit Codes:
    0 - Always (results are reported, not fatal)

Output:
    Performance table with cold start (includes JIT) and warm timings.
"""

import argparse
import gc
import json
import os
import platform
import sys
import time
from datetime import datetime

import numpy as np

# Add parent directory for imports
_this_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_this_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Check ROOT availability
try:
    import ROOT
    HAS_ROOT = True
    ROOT_VERSION = ROOT.gROOT.GetVersion()
except ImportError:
    HAS_ROOT = False
    ROOT_VERSION = None

from AliasDataFrame import AliasDataFrame


# =============================================================================
# Benchmark Functions
# =============================================================================

def benchmark_rdf(tree, aDF, aliases, n_iterations=5, enable_mt=False):
    """
    Benchmark RDataFrame approach.
    
    Parameters
    ----------
    tree : ROOT.TTree
        Pre-loaded tree with friends attached
    aDF : AliasDataFrame
        AliasDataFrame with schema
    aliases : list of str
        Aliases to evaluate
    n_iterations : int
        Number of iterations (first is cold, rest are warm)
    enable_mt : bool
        Enable multi-threading
        
    Returns
    -------
    dict
        Timing results
    """
    from AliasDataFrameRDF import get_ordered_defines
    
    if enable_mt:
        ROOT.EnableImplicitMT()
    else:
        ROOT.DisableImplicitMT()
    
    times = []
    entries = 0
    
    for i in range(n_iterations):
        gc.collect()
        
        df = ROOT.RDataFrame(tree)
        defines = get_ordered_defines(aliases, aDF=aDF)
        
        t0 = time.perf_counter()
        
        for d in defines:
            df = df.Define(d['name'], d['cpp_expr'])
        
        # Force evaluation with histogram
        h = df.Histo1D(("h", "", 100, -10, 10), aliases[0])
        entries = int(h.GetEntries())
        
        times.append(time.perf_counter() - t0)
    
    n_threads = ROOT.GetThreadPoolSize() if enable_mt else 1
    
    if enable_mt:
        ROOT.DisableImplicitMT()
    
    return {
        'cold': times[0],
        'warm_min': min(times[1:]) if len(times) > 1 else times[0],
        'warm_mean': np.mean(times[1:]) if len(times) > 1 else times[0],
        'warm_std': np.std(times[1:]) if len(times) > 1 else 0,
        'entries': entries,
        'n_threads': n_threads,
        'defines': defines,  # Include for debugging
    }


def benchmark_ttree_draw(tree, expression, n_iterations=5):
    """
    Benchmark TTree::Draw approach.
    
    Parameters
    ----------
    tree : ROOT.TTree
        Pre-loaded tree with friends attached
    expression : str
        Expression to draw (should be C++ compatible)
    n_iterations : int
        Number of iterations
        
    Returns
    -------
    dict
        Timing results
    """
    times = []
    entries = 0
    
    for i in range(n_iterations):
        gc.collect()
        
        t0 = time.perf_counter()
        entries = tree.Draw(expression, "", "goff")
        times.append(time.perf_counter() - t0)
    
    return {
        'cold': times[0],
        'warm_min': min(times[1:]) if len(times) > 1 else times[0],
        'warm_mean': np.mean(times[1:]) if len(times) > 1 else times[0],
        'warm_std': np.std(times[1:]) if len(times) > 1 else 0,
        'entries': entries,
    }


def benchmark_aliasdf(filepath, aliases, n_iterations=5):
    """
    Benchmark AliasDataFrame materialize approach.
    
    Parameters
    ----------
    filepath : str
        Path to ROOT file
    aliases : list of str
        Aliases to evaluate
    n_iterations : int
        Number of iterations
        
    Returns
    -------
    dict
        Timing results
    """
    times = []
    entries = 0
    
    for i in range(n_iterations):
        gc.collect()
        
        # Fresh load each iteration (measuring computation, data is cached by OS)
        aDF = AliasDataFrame.read_tree(filepath, "tree", load_subframes=True)
        
        t0 = time.perf_counter()
        aDF.materialize_aliases(names=aliases, with_dependencies=True)
        times.append(time.perf_counter() - t0)
        
        entries = len(aDF.df)
    
    return {
        'cold': times[0],
        'warm_min': min(times[1:]) if len(times) > 1 else times[0],
        'warm_mean': np.mean(times[1:]) if len(times) > 1 else times[0],
        'warm_std': np.std(times[1:]) if len(times) > 1 else 0,
        'entries': entries,
    }


# =============================================================================
# Validation Functions
# =============================================================================

def validate_correctness(tree, aDF, rdf_alias, mat_column, rtol=1e-5):
    """
    Compare RDataFrame-computed alias against pre-materialized ground truth.
    
    Parameters
    ----------
    tree : ROOT.TTree
        Tree with friends attached
    aDF : AliasDataFrame
        AliasDataFrame with schema for get_ordered_defines
    rdf_alias : str
        Alias name to compute via RDataFrame (e.g., "dyC2")
    mat_column : str
        Ground truth column name (e.g., "dyC2_mat")
    rtol : float
        Relative tolerance for comparison
        
    Returns
    -------
    bool
        True if validation passed
    """
    from AliasDataFrameRDF import get_ordered_defines
    
    print(f"\n[VALIDATION] {rdf_alias} vs {mat_column}")
    
    # Check if ground truth column exists
    if not tree.GetBranch(mat_column):
        print(f"  ⚠ Ground truth column '{mat_column}' not found in tree")
        print(f"  SKIPPED")
        return None
    
    try:
        df = ROOT.RDataFrame(tree)
        
        # Apply defines for the alias
        defines = get_ordered_defines([rdf_alias], aDF=aDF)
        for d in defines:
            df = df.Define(d['name'], d['cpp_expr'])
        
        # Get RDataFrame result
        rdf_values = np.asarray(df.AsNumpy([rdf_alias])[rdf_alias])
        
        # Get ground truth
        mat_values = np.asarray(df.AsNumpy([mat_column])[mat_column])
        
        # Compute statistics
        rdf_mean = np.nanmean(rdf_values)
        mat_mean = np.nanmean(mat_values)
        rdf_std = np.nanstd(rdf_values)
        mat_std = np.nanstd(mat_values)
        
        print(f"  RDataFrame:   mean={rdf_mean:.6f}, std={rdf_std:.6f}")
        print(f"  Ground truth: mean={mat_mean:.6f}, std={mat_std:.6f}")
        
        # Compare
        mean_ok = np.isclose(rdf_mean, mat_mean, rtol=rtol, equal_nan=True)
        std_ok = np.isclose(rdf_std, mat_std, rtol=rtol, equal_nan=True)
        
        if mean_ok and std_ok:
            print(f"  ✅ PASS")
            return True
        else:
            # Compute more detailed comparison
            diff = rdf_values - mat_values
            max_diff = np.nanmax(np.abs(diff))
            print(f"  Max absolute diff: {max_diff:.6e}")
            print(f"  ❌ FAIL - results differ!")
            return False
            
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        return False


# =============================================================================
# Output Functions
# =============================================================================

def print_results(results, baseline='TTree::Draw', expression=None, aliases=None):
    """Print formatted benchmark results."""
    
    print("\n" + "="*70)
    print("BENCHMARK RESULTS")
    print("="*70)
    
    # Show what was benchmarked
    if expression:
        print(f"\nExpression: {expression}")
    if aliases:
        print(f"Aliases: {aliases}")
    
    # Header
    print(f"\n{'Method':<25} | {'Cold (s)':<10} | {'Warm (s)':<10} | {'Speedup':<10}")
    print("-"*70)
    
    baseline_warm = results.get(baseline, {}).get('warm_mean', 1.0)
    if baseline_warm == 0:
        baseline_warm = 1.0
    
    for name, data in results.items():
        cold = data['cold']
        warm = data['warm_mean']
        speedup = baseline_warm / warm if warm > 0 else 0
        
        suffix = ""
        if 'n_threads' in data and data['n_threads'] > 1:
            suffix = f" ({data['n_threads']}T)"
        
        print(f"{name + suffix:<25} | {cold:<10.3f} | {warm:<10.3f} | {speedup:<10.2f}x")
    
    print("-"*70)
    
    # Get entries from first result
    entries = next(iter(results.values()), {}).get('entries', 'N/A')
    if isinstance(entries, int):
        entries = f"{entries:,}"
    print(f"\nEntries processed: {entries}")
    print(f"Baseline: {baseline}")


def export_json(results, filepath, n_rows, mode, aliases):
    """
    Export results to JSON file.
    
    Matches structure from other benchmarks for compatibility.
    """
    # Create directory if needed
    dirpath = os.path.dirname(filepath)
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)
    
    # Build metrics dict for compatibility with baseline_utils
    metrics = {}
    for name, data in results.items():
        key = name.lower().replace(' ', '_').replace('::', '_')
        metrics[f'{key}_cold'] = data['cold']
        metrics[f'{key}_warm'] = data['warm_mean']
        if 'n_threads' in data:
            metrics[f'{key}_threads'] = data['n_threads']
    
    # Add speedup metrics
    baseline_warm = results.get('TTree::Draw', results.get('RDataFrame', {})).get('warm_mean', 1.0)
    if baseline_warm and baseline_warm > 0:
        for name, data in results.items():
            if data['warm_mean'] > 0:
                key = name.lower().replace(' ', '_').replace('::', '_')
                metrics[f'{key}_speedup'] = baseline_warm / data['warm_mean']
    
    output = {
        'benchmark': 'benchmark_rdf.py',
        'timestamp': datetime.now().isoformat(),
        'hostname': platform.node(),
        'python_version': platform.python_version(),
        'platform': platform.platform(),
        'root_version': ROOT_VERSION,
        'rows': n_rows,
        'mode': mode,
        'aliases': aliases,
        'time_s': sum(r['warm_mean'] for r in results.values()),
        'metrics': metrics,
        'results': results,
    }
    
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nResults exported to: {filepath}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Benchmark RDF vs TTree::Draw vs AliasDataFrame',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # List available columns
    python benchmark_rdf.py rdf_benchmark.root --list
    
    # Benchmark raw columns (always available)
    python benchmark_rdf.py rdf_benchmark.root --aliases y z
    python benchmark_rdf.py rdf_benchmark.root --aliases dy dz --mt
    
    # Benchmark with aliases (requires schema or calibration data)
    python benchmark_rdf.py calibration.root --aliases dyC2 dzC2
    
    # Validate RDataFrame results against ground truth
    python benchmark_rdf.py rdf_11M.root --aliases L10 --validate
    python benchmark_rdf.py rdf_11M.root --aliases L20 --validate --iterations 3
    
    # Export to JSON
    python benchmark_rdf.py data.root --json results/benchmark_rdf.json
        """
    )
    parser.add_argument('filepath', help='Path to ROOT file')
    parser.add_argument('--aliases', nargs='+', default=['y'], 
                        help='Columns or aliases to benchmark (default: y)')
    parser.add_argument('--iterations', type=int, default=5,
                        help='Number of iterations (default: 5)')
    parser.add_argument('--mt', action='store_true',
                        help='Also benchmark RDataFrame with multi-threading')
    parser.add_argument('--skip-ttree', action='store_true',
                        help='Skip TTree::Draw benchmark')
    parser.add_argument('--skip-adf', action='store_true',
                        help='Skip AliasDataFrame benchmark')
    parser.add_argument('--json', type=str, metavar='FILE',
                        help='Export results to JSON file')
    parser.add_argument('--json-output', type=str, metavar='FILE',
                        help='Save results to JSON file (alias for --json)')
    parser.add_argument('--quiet', action='store_true',
                        help='Minimal output')
    parser.add_argument('--list', action='store_true',
                        help='List available columns and aliases, then exit')
    parser.add_argument('--validate', action='store_true',
                        help='Validate RDataFrame results against ground truth (_mat columns)')
    
    args = parser.parse_args()
    
    # Validate file exists
    if not os.path.exists(args.filepath):
        print(f"ERROR: File not found: {args.filepath}")
        return 1
    
    if not HAS_ROOT:
        print("ERROR: ROOT is required for this benchmark")
        print("Skipping benchmark_rdf.py")
        return 0  # Don't fail, just skip
    
    verbose = not args.quiet
    
    if verbose:
        print("="*70)
        print("BENCHMARK: RDataFrame vs TTree::Draw vs AliasDataFrame")
        print("="*70)
        print(f"File: {args.filepath}")
        print(f"Aliases: {args.aliases}")
        print(f"Iterations: {args.iterations}")
        print(f"ROOT version: {ROOT_VERSION}")
        print(f"Hostname: {platform.node()}")
        print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Load data
    if verbose:
        print("\nLoading data...")
    
    try:
        from AliasDataFrameRDF import setup_tree_with_friends, get_ordered_defines
        
        adf = AliasDataFrame.read_tree(args.filepath, "tree", load_subframes=True)
        
        # Get available columns and aliases
        available_columns = list(adf.df.columns) if hasattr(adf, 'df') else []
        available_aliases = list(adf.aliases.keys()) if hasattr(adf, 'aliases') else []
        
        # Handle --list flag
        if args.list:
            print(f"\nFile: {args.filepath}")
            print(f"\nColumns ({len(available_columns)}):")
            for col in available_columns:
                print(f"  {col}")
            if available_aliases:
                print(f"\nAliases ({len(available_aliases)}):")
                for alias in available_aliases:
                    expr = adf.aliases.get(alias, '')
                    print(f"  {alias} = {expr}")
            else:
                print("\nNo aliases defined in this file.")
            return 0
        
        # Validate requested items exist
        all_available = set(available_columns) | set(available_aliases)
        missing = [a for a in args.aliases if a not in all_available]
        
        if missing:
            print(f"\n⚠ Warning: Not found: {missing}")
            print(f"  Available columns: {available_columns}")
            if available_aliases:
                print(f"  Available aliases: {available_aliases}")
            
            fallback = available_columns[0] if available_columns else 'y'
            print(f"\n  Falling back to: {fallback}")
            args.aliases = [fallback]
        
        tree, f = setup_tree_with_friends(args.filepath, "tree", adf.schema)
        
        n_rows = tree.GetEntries()
        if verbose:
            print(f"Entries: {n_rows:,}")
            if available_aliases:
                print(f"Aliases: {available_aliases}")
            
            # Show what defines will be generated
            defines = get_ordered_defines(args.aliases, aDF=adf)
            if defines:
                print(f"\nDefines for RDataFrame ({len(defines)}):")
                for d in defines:
                    print(f"  {d['name']} = {d['cpp_expr']}")
            
    except Exception as e:
        print(f"ERROR: Failed to load data: {e}")
        import traceback
        traceback.print_exc()
        print("Skipping benchmark_rdf.py")
        return 0
    
    results = {}
    mode = 'full' if args.iterations >= 5 else 'quick'
    
    # TTree::Draw benchmark
    if not args.skip_ttree:
        if verbose:
            print("\nBenchmarking TTree::Draw...")
        ttree_expr = args.aliases[0]
        try:
            results['TTree::Draw'] = benchmark_ttree_draw(tree, ttree_expr, args.iterations)
            if verbose:
                print(f"  Cold: {results['TTree::Draw']['cold']:.3f}s, Warm: {results['TTree::Draw']['warm_mean']:.3f}s")
        except Exception as e:
            if verbose:
                print(f"  SKIPPED: {e}")
    
    # RDataFrame benchmark (single-thread)
    if verbose:
        print("\nBenchmarking RDataFrame (1 thread)...")
    try:
        results['RDataFrame'] = benchmark_rdf(tree, adf, args.aliases, args.iterations, enable_mt=False)
        if verbose:
            print(f"  Cold: {results['RDataFrame']['cold']:.3f}s, Warm: {results['RDataFrame']['warm_mean']:.3f}s")
    except Exception as e:
        if verbose:
            print(f"  SKIPPED: {e}")
    
    # RDataFrame benchmark (multi-thread)
    if args.mt:
        if verbose:
            print("\nBenchmarking RDataFrame (MT)...")
        try:
            results['RDataFrame MT'] = benchmark_rdf(tree, adf, args.aliases, args.iterations, enable_mt=True)
            if verbose:
                print(f"  Cold: {results['RDataFrame MT']['cold']:.3f}s, Warm: {results['RDataFrame MT']['warm_mean']:.3f}s")
                print(f"  Threads: {results['RDataFrame MT']['n_threads']}")
        except Exception as e:
            if verbose:
                print(f"  SKIPPED: {e}")
    
    # AliasDataFrame benchmark
    if not args.skip_adf:
        if verbose:
            print("\nBenchmarking AliasDataFrame...")
        try:
            results['AliasDataFrame'] = benchmark_aliasdf(args.filepath, args.aliases, args.iterations)
            if verbose:
                print(f"  Cold: {results['AliasDataFrame']['cold']:.3f}s, Warm: {results['AliasDataFrame']['warm_mean']:.3f}s")
        except Exception as e:
            if verbose:
                print(f"  SKIPPED: {e}")
    
    # Validation (if requested)
    validation_results = {}
    if args.validate:
        if verbose:
            print("\n" + "="*70)
            print("VALIDATION")
            print("="*70)
        
        for alias in args.aliases:
            mat_column = f"{alias}_mat"
            result = validate_correctness(tree, adf, alias, mat_column)
            validation_results[alias] = result
    
    # Print summary
    if results:
        baseline = 'RDataFrame' if args.skip_ttree or 'TTree::Draw' not in results else 'TTree::Draw'
        
        # Get expression info from RDF results
        rdf_defines = None
        if 'RDataFrame' in results and 'defines' in results['RDataFrame']:
            rdf_defines = results['RDataFrame']['defines']
        elif 'RDataFrame MT' in results and 'defines' in results['RDataFrame MT']:
            rdf_defines = results['RDataFrame MT']['defines']
        
        # Build expression string for display
        if rdf_defines and len(rdf_defines) > 0:
            expr_strs = [f"{d['name']} = {d['cpp_expr']}" for d in rdf_defines]
            expression = "; ".join(expr_strs)
        else:
            expression = f"{args.aliases[0]} (raw column, no alias expression)"
        
        if verbose:
            print_results(results, baseline=baseline, expression=expression, aliases=args.aliases)
            
            # Warnings and notes
            if not rdf_defines or len(rdf_defines) == 0:
                print(f"\n⚠ Warning: '{args.aliases[0]}' is a raw column, not an alias.")
                print(f"  RDataFrame has JIT overhead with no computation benefit.")
                print(f"  Try an alias with an expression (e.g., --aliases dyC2)")
            
            if n_rows < 10_000_000:
                print(f"\nNote: RDataFrame JIT overhead dominates at {n_rows:,} rows.")
                print(f"      Regenerate with: python generate_synthetic_data.py --rdf --rows 10000000")
            
            # Print validation summary
            if args.validate and validation_results:
                passed = sum(1 for v in validation_results.values() if v is True)
                failed = sum(1 for v in validation_results.values() if v is False)
                skipped = sum(1 for v in validation_results.values() if v is None)
                
                print(f"\nValidation: ", end="")
                if failed == 0 and passed > 0:
                    print(f"✅ PASSED ({passed}/{passed + skipped})")
                elif failed > 0:
                    print(f"❌ FAILED ({failed} failed, {passed} passed)")
                else:
                    print(f"⚠ SKIPPED (no ground truth columns found)")
    else:
        print("\nNo benchmark results collected.")
    
    # Export to JSON if requested (support both --json and --json-output)
    json_file = args.json or args.json_output
    if json_file and results:
        # Compute validation status
        all_valid = all(v is True for v in validation_results.values()) if validation_results else None
        
        # Compute baseline for speedup
        baseline_warm = results.get('TTree::Draw', results.get('RDataFrame', {})).get('warm_mean', 1.0)
        if baseline_warm == 0:
            baseline_warm = 1.0
        
        output = {
            'timestamp': datetime.now().isoformat(),
            'file': args.filepath,
            'entries': n_rows,
            'aliases': args.aliases,
            'validation_passed': all_valid,
            'results': {
                name: {
                    'cold': data['cold'],
                    'warm': data['warm_mean'],
                    'speedup': baseline_warm / data['warm_mean'] if data['warm_mean'] > 0 else 0
                }
                for name, data in results.items()
            }
        }
        
        # Create directory if needed
        json_dir = os.path.dirname(json_file)
        if json_dir and not os.path.exists(json_dir):
            os.makedirs(json_dir, exist_ok=True)
        
        with open(json_file, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {json_file}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
