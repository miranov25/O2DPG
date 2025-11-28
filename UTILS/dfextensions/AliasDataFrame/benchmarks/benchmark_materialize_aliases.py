#!/usr/bin/env python3
"""
benchmark_materialize_aliases.py - Benchmark materialize_aliases() performance

Tests alias materialization with realistic DAG complexity and subframe joins.
Measures performance difference between fill_mode='safe' and fill_mode='direct'.

Usage:
    python benchmark_materialize_aliases.py                    # Full benchmark (500k rows)
    python benchmark_materialize_aliases.py --quick            # Quick mode (100k rows)
    python benchmark_materialize_aliases.py --json results.json
    python benchmark_materialize_aliases.py --rows 1000000     # Custom row count

Exit Codes:
    0 - Always (results are reported, not fatal)

Scenarios:
    1. simple:  Alias chain without subframes (baseline)
    2. safe:    Subframe joins with fill_mode='safe' (full NaN/Inf processing)
    3. direct:  Subframe joins with fill_mode='direct' (skip NaN/Inf checks)
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

# Row counts
DEFAULT_ROWS = 500_000      # ~3s target runtime
QUICK_ROWS = 100_000        # CI/quick mode

# Random seed for reproducibility
RNG_SEED = 12345

# Default dtype for physics columns
DEFAULT_DTYPE = np.float32

# Subframe coverage configuration
# Simulates real case: ~5% missing (rows > 160 have no calibration)
ROW_MAX_WITH_CALIBRATION = 160
ROW_TOTAL = 190


# =============================================================================
# Synthetic Data Generation
# =============================================================================

def create_main_df(n_rows, rng):
    """
    Create main DataFrame with ITSTPC-like schema.
    
    Schema mimics TPC calibration data with:
    - drift25: drift region (0-3)
    - side: detector side (0-1)
    - row: pad row (0-190), where >160 has no calibration
    - r, phi: polar coordinates
    - y2x: track angle
    - dyC1, dzC1: residuals to calibrate
    
    Parameters
    ----------
    n_rows : int
        Number of rows
    rng : numpy.random.Generator
        Random number generator
        
    Returns
    -------
    pd.DataFrame
    """
    # Key columns for subframe join
    drift25 = rng.integers(0, 4, size=n_rows, dtype=np.int8)
    side = rng.integers(0, 2, size=n_rows, dtype=np.int8)
    row = rng.integers(0, ROW_TOTAL, size=n_rows, dtype=np.int16)
    
    # Physics columns
    r = rng.uniform(80.0, 250.0, size=n_rows).astype(DEFAULT_DTYPE)
    phi = rng.uniform(0.0, 2 * np.pi, size=n_rows).astype(DEFAULT_DTYPE)
    y2x = rng.normal(0.0, 0.2, size=n_rows).astype(DEFAULT_DTYPE)
    
    # Residuals to calibrate
    dyC1 = rng.normal(0.0, 0.5, size=n_rows).astype(DEFAULT_DTYPE)
    dzC1 = rng.normal(0.0, 0.5, size=n_rows).astype(DEFAULT_DTYPE)
    
    return pd.DataFrame({
        "drift25": drift25,
        "side": side,
        "row": row,
        "r": r,
        "phi": phi,
        "y2x": y2x,
        "dyC1": dyC1,
        "dzC1": dzC1,
    })


def create_subframe_df(rng):
    """
    Create subframe DataFrame with calibration coefficients.
    
    Simulates DITS0FitSide calibration table:
    - Keys: (drift25, side, row)
    - Only rows 0-160 have calibration (~5% missing)
    - Contains intercept and slope coefficients
    
    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator
        
    Returns
    -------
    pd.DataFrame
    """
    # Generate all valid key combinations
    # drift25: 0-3, side: 0-1, row: 0-160 (no calibration for row > 160)
    keys = []
    for d in range(4):
        for s in range(2):
            for r in range(ROW_MAX_WITH_CALIBRATION + 1):  # 0-160 inclusive
                keys.append((d, s, r))
    
    n_keys = len(keys)
    
    # Extract key columns
    drift25 = np.array([k[0] for k in keys], dtype=np.int8)
    side = np.array([k[1] for k in keys], dtype=np.int8)
    row = np.array([k[2] for k in keys], dtype=np.int16)
    
    # Calibration coefficients (realistic ranges)
    dyC1_intercept = rng.normal(0.0, 0.2, n_keys).astype(DEFAULT_DTYPE)
    dzC1_intercept = rng.normal(0.0, 0.2, n_keys).astype(DEFAULT_DTYPE)
    dyC1_slope_rrel = rng.normal(0.0, 0.01, n_keys).astype(DEFAULT_DTYPE)
    dzC1_slope_rrel = rng.normal(0.0, 0.01, n_keys).astype(DEFAULT_DTYPE)
    dyC1_slope_y2x = rng.normal(0.0, 0.1, n_keys).astype(DEFAULT_DTYPE)
    dzC1_slope_y2x = rng.normal(0.0, 0.1, n_keys).astype(DEFAULT_DTYPE)
    
    # Additional coefficients for more complex DAG
    dyC1_slope_phi = rng.normal(0.0, 0.05, n_keys).astype(DEFAULT_DTYPE)
    dzC1_slope_phi = rng.normal(0.0, 0.05, n_keys).astype(DEFAULT_DTYPE)
    
    return pd.DataFrame({
        "drift25": drift25,
        "side": side,
        "row": row,
        "dyC1_intercept": dyC1_intercept,
        "dzC1_intercept": dzC1_intercept,
        "dyC1_slope_rrel": dyC1_slope_rrel,
        "dzC1_slope_rrel": dzC1_slope_rrel,
        "dyC1_slope_y2x": dyC1_slope_y2x,
        "dzC1_slope_y2x": dzC1_slope_y2x,
        "dyC1_slope_phi": dyC1_slope_phi,
        "dzC1_slope_phi": dzC1_slope_phi,
    })


# =============================================================================
# Alias Definitions
# =============================================================================

def add_simple_aliases(adf):
    """
    Add simple alias chain (no subframes).
    
    Used for Scenario 1 to measure baseline alias overhead.
    
    Returns
    -------
    list : Names of target aliases
    """
    # Layer A: Geometry transformations
    adf.add_alias('rrel', 'r - 165.0')  # Relative radius (center of TPC)
    adf.add_alias('rrel2', 'rrel * rrel')
    adf.add_alias('cosPhi', 'cos(phi)')
    adf.add_alias('sinPhi', 'sin(phi)')
    adf.add_alias('y2x2', 'y2x * y2x')
    
    # Layer B: Intermediate computations
    adf.add_alias('rrel_scaled', 'rrel * 0.01')
    adf.add_alias('y2x_scaled', 'y2x * 0.1')
    adf.add_alias('phi_term', 'cosPhi * sinPhi')
    
    # Layer C: Mixed dependencies
    adf.add_alias('geom_term1', 'rrel_scaled + y2x_scaled')
    adf.add_alias('geom_term2', 'rrel2 * 0.0001 + phi_term')
    
    # Layer D: Final outputs
    adf.add_alias('simple_result', 'geom_term1 + geom_term2')
    
    return ['simple_result']


def add_subframe_aliases(adf):
    """
    Add full alias DAG with subframe dependencies.
    
    Used for Scenarios 2 and 3 to measure subframe join overhead.
    Creates ~35 aliases in a multi-layer DAG.
    
    Returns
    -------
    list : Names of target aliases (final outputs)
    """
    # =========================================================================
    # Layer A: Geometry aliases (main-frame only)
    # =========================================================================
    adf.add_alias('rrel', 'r - 165.0')
    adf.add_alias('rrel2', 'rrel * rrel')
    adf.add_alias('cosPhi', 'cos(phi)')
    adf.add_alias('sinPhi', 'sin(phi)')
    adf.add_alias('tanPhi', 'sinPhi / (cosPhi + 1e-10)')  # Avoid div by zero
    adf.add_alias('abs_y2x', 'abs(y2x)')
    adf.add_alias('y2x2', 'y2x * y2x')
    adf.add_alias('phi_mod', 'phi - floor(phi / (2 * 3.14159)) * 2 * 3.14159')
    
    # =========================================================================
    # Layer B: Subframe-based projections (calibration corrections)
    # Uses DITS0FitSide.column syntax for subframe access
    # =========================================================================
    
    # Y residual correction from calibration
    adf.add_alias('dyC1_SC', 
        'DITS0FitSide.dyC1_intercept + '
        'rrel * DITS0FitSide.dyC1_slope_rrel + '
        'y2x * DITS0FitSide.dyC1_slope_y2x + '
        'cosPhi * DITS0FitSide.dyC1_slope_phi'
    )
    
    # Z residual correction from calibration
    adf.add_alias('dzC1_SC', 
        'DITS0FitSide.dzC1_intercept + '
        'rrel * DITS0FitSide.dzC1_slope_rrel + '
        'y2x * DITS0FitSide.dzC1_slope_y2x + '
        'sinPhi * DITS0FitSide.dzC1_slope_phi'
    )
    
    # =========================================================================
    # Layer C: Intermediate corrections (mixed geometry + subframe)
    # =========================================================================
    
    # Quadratic radius correction
    adf.add_alias('dyC1_SC_r2', 'dyC1_SC + 0.0001 * rrel2')
    adf.add_alias('dzC1_SC_r2', 'dzC1_SC + 0.0001 * rrel2')
    
    # y2x squared correction
    adf.add_alias('dyC1_SC_y2x2', 'dyC1_SC + 0.05 * y2x2')
    adf.add_alias('dzC1_SC_y2x2', 'dzC1_SC + 0.05 * y2x2')
    
    # Combined corrections
    adf.add_alias('dyC1_SC_combined', '(dyC1_SC_r2 + dyC1_SC_y2x2) * 0.5')
    adf.add_alias('dzC1_SC_combined', '(dzC1_SC_r2 + dzC1_SC_y2x2) * 0.5')
    
    # Phi-dependent modulation
    adf.add_alias('dy_phi_mod', 'dyC1_SC_combined * cosPhi')
    adf.add_alias('dz_phi_mod', 'dzC1_SC_combined * sinPhi')
    
    # Cross-terms
    adf.add_alias('dy_cross', 'dy_phi_mod + 0.1 * tanPhi')
    adf.add_alias('dz_cross', 'dz_phi_mod + 0.1 * abs_y2x')
    
    # =========================================================================
    # Layer D: Final calibration and residuals
    # =========================================================================
    
    # Final calibration values
    adf.add_alias('dyC1_CalibAll', 'dyC1_SC_combined + dy_cross * 0.01')
    adf.add_alias('dzC1_CalibAll', 'dzC1_SC_combined + dz_cross * 0.01')
    
    # Corrected residuals (target outputs)
    adf.add_alias('dyC2', 'dyC1 - dyC1_CalibAll')
    adf.add_alias('dzC2', 'dzC1 - dzC1_CalibAll')
    
    # Additional derived quantities
    adf.add_alias('dC2_magnitude', 'sqrt(dyC2 * dyC2 + dzC2 * dzC2)')
    adf.add_alias('dC2_ratio', 'dyC2 / (dzC2 + 1e-10)')
    
    return ['dyC2', 'dzC2']


# =============================================================================
# Measurement Utilities
# =============================================================================

def measure_materialize(fn, adf):
    """
    Measure time and memory for a materialization operation.
    
    Parameters
    ----------
    fn : callable
        Function to measure (should call materialize_aliases)
    adf : AliasDataFrame
        The AliasDataFrame being operated on
        
    Returns
    -------
    dict : {time_s, rows_per_sec, peak_mb, df_mem_mb}
    """
    gc.collect()
    tracemalloc.start()
    
    t0 = time.perf_counter()
    fn()
    elapsed = time.perf_counter() - t0
    
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    n_rows = len(adf.df)
    df_mem = adf.df.memory_usage(deep=True).sum()
    
    return {
        'time_s': elapsed,
        'rows_per_sec': n_rows / elapsed if elapsed > 0 else float('inf'),
        'peak_mb': peak / (1024 * 1024),
        'df_mem_mb': df_mem / (1024 * 1024),
    }


# =============================================================================
# Benchmark Scenarios
# =============================================================================

def run_scenario_simple(df_main, verbose=True):
    """
    Scenario 1: Simple alias chain without subframes.
    
    Measures baseline alias materialization overhead.
    """
    if verbose:
        print("\n--- Scenario 1: Simple (no subframe) ---")
    
    adf = AliasDataFrame(df_main.copy())
    targets = add_simple_aliases(adf)
    n_aliases = len(adf.aliases)
    
    if verbose:
        print(f"  Aliases defined: {n_aliases}")
        print(f"  Targets: {targets}")
    
    def do_materialize():
        adf.materialize_aliases(
            names=targets,
            with_dependencies=True,
            cleanTemporary=True,
        )
    
    result = measure_materialize(do_materialize, adf)
    result['n_aliases'] = n_aliases
    result['n_targets'] = len(targets)
    
    if verbose:
        print(f"  Time: {result['time_s']:.3f}s")
        print(f"  Rows/sec: {result['rows_per_sec']:,.0f}")
        print(f"  Peak memory: {result['peak_mb']:.1f} MB")
    
    return result


def run_scenario_subframe(df_main, df_subframe, fill_mode, verbose=True):
    """
    Scenario 2/3: Subframe joins with fill_mode configuration.
    
    Parameters
    ----------
    df_main : pd.DataFrame
        Main DataFrame
    df_subframe : pd.DataFrame
        Subframe DataFrame with calibration coefficients
    fill_mode : str
        'safe' or 'direct'
    verbose : bool
        Print progress
        
    Returns
    -------
    dict : Benchmark results
    """
    mode_name = fill_mode.capitalize()
    if verbose:
        print(f"\n--- Scenario: Subframe ({mode_name}) ---")
    
    # Create fresh AliasDataFrame
    adf = AliasDataFrame(df_main.copy())
    
    # Register subframe with multi-key join
    adf.register_subframe(
        'DITS0FitSide',
        AliasDataFrame(df_subframe.copy()),
        index_columns=['drift25', 'side', 'row'],
    )
    
    # Configure fill mode
    if fill_mode == 'safe':
        adf.set_subframe_fill(
            'DITS0FitSide',
            fill_missing=None,      # Keep NaN for missing
            fill_invalid=None,      # Keep NaN/Inf
            warn_missing_keys=False,
            fill_mode='safe',
        )
    elif fill_mode == 'direct':
        adf.set_subframe_fill(
            'DITS0FitSide',
            fill_missing=0.0,       # Fill missing with 0
            fill_invalid=0.0,       # Fill NaN/Inf with 0
            warn_missing_keys=False,
            fill_mode='direct',
        )
    
    # Add aliases
    targets = add_subframe_aliases(adf)
    n_aliases = len(adf.aliases)
    
    if verbose:
        print(f"  Aliases defined: {n_aliases}")
        print(f"  Targets: {targets}")
        print(f"  Fill mode: {fill_mode}")
    
    def do_materialize():
        adf.materialize_aliases(
            names=targets,
            with_dependencies=True,
            cleanTemporary=True,
        )
    
    result = measure_materialize(do_materialize, adf)
    result['n_aliases'] = n_aliases
    result['n_targets'] = len(targets)
    result['fill_mode'] = fill_mode
    
    # Calculate missing key statistics
    n_missing = (adf.df['row'] > ROW_MAX_WITH_CALIBRATION).sum()
    result['missing_keys_pct'] = 100.0 * n_missing / len(adf.df)
    
    if verbose:
        print(f"  Time: {result['time_s']:.3f}s")
        print(f"  Rows/sec: {result['rows_per_sec']:,.0f}")
        print(f"  Peak memory: {result['peak_mb']:.1f} MB")
        print(f"  Missing keys: {result['missing_keys_pct']:.1f}%")
    
    return result


# =============================================================================
# Main Benchmark Runner
# =============================================================================

def run_all_benchmarks(n_rows, verbose=True):
    """
    Run all benchmark scenarios.
    
    Returns
    -------
    dict : All results
    """
    if verbose:
        print("=" * 60)
        print("MATERIALIZE_ALIASES BENCHMARK")
        print("=" * 60)
        print(f"Rows:      {n_rows:,}")
        print(f"Hostname:  {platform.node()}")
        print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Initialize RNG
    rng = np.random.default_rng(RNG_SEED)
    
    # Generate data
    if verbose:
        print("\nGenerating synthetic data...")
    
    t0 = time.perf_counter()
    df_main = create_main_df(n_rows, rng)
    df_subframe = create_subframe_df(rng)
    data_gen_time = time.perf_counter() - t0
    
    if verbose:
        print(f"  Main DataFrame: {len(df_main):,} rows × {len(df_main.columns)} cols")
        print(f"  Subframe: {len(df_subframe):,} rows × {len(df_subframe.columns)} cols")
        print(f"  Generation time: {data_gen_time:.2f}s")
        
        # Calculate expected missing percentage
        n_missing = (df_main['row'] > ROW_MAX_WITH_CALIBRATION).sum()
        pct_missing = 100.0 * n_missing / len(df_main)
        print(f"  Expected missing: {pct_missing:.1f}% (row > {ROW_MAX_WITH_CALIBRATION})")
    
    results = {}
    
    # Scenario 1: Simple (no subframe)
    results['simple'] = run_scenario_simple(df_main, verbose)
    
    # Scenario 2: Subframe with fill_mode='safe'
    results['safe'] = run_scenario_subframe(df_main, df_subframe, 'safe', verbose)
    
    # Scenario 3: Subframe with fill_mode='direct'
    results['direct'] = run_scenario_subframe(df_main, df_subframe, 'direct', verbose)
    
    # Calculate speedup
    if results['safe']['time_s'] > 0 and results['direct']['time_s'] > 0:
        results['direct_vs_safe_speedup'] = (
            results['safe']['time_s'] / results['direct']['time_s']
        )
    else:
        results['direct_vs_safe_speedup'] = None
    
    # Calculate subframe vs simple overhead
    if results['simple']['time_s'] > 0:
        results['safe_vs_simple_ratio'] = (
            results['safe']['time_s'] / results['simple']['time_s']
        )
        results['direct_vs_simple_ratio'] = (
            results['direct']['time_s'] / results['simple']['time_s']
        )
    
    return results


def print_summary(results, n_rows):
    """Print benchmark summary table."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print(f"\n{'Scenario':<15} {'Time (s)':<12} {'Rows/sec':<15} {'Aliases':<10}")
    print("-" * 60)
    
    for name in ['simple', 'safe', 'direct']:
        r = results[name]
        rows_sec = f"{r['rows_per_sec']:,.0f}"
        print(f"{name:<15} {r['time_s']:<12.3f} {rows_sec:<15} {r['n_aliases']:<10}")
    
    print("-" * 60)
    
    # Total time
    total_time = sum(results[s]['time_s'] for s in ['simple', 'safe', 'direct'])
    print(f"{'Total':<15} {total_time:<12.3f}")
    
    # Speedup metrics
    print("\n" + "-" * 60)
    print("SPEEDUP METRICS")
    print("-" * 60)
    
    if results.get('direct_vs_safe_speedup'):
        speedup = results['direct_vs_safe_speedup']
        print(f"  direct vs safe:   {speedup:.2f}x {'(faster)' if speedup > 1 else '(slower)'}")
    
    if results.get('safe_vs_simple_ratio'):
        ratio = results['safe_vs_simple_ratio']
        print(f"  safe vs simple:   {ratio:.2f}x (subframe overhead)")
    
    if results.get('direct_vs_simple_ratio'):
        ratio = results['direct_vs_simple_ratio']
        print(f"  direct vs simple: {ratio:.2f}x (subframe overhead)")
    
    # Missing key stats
    if 'missing_keys_pct' in results['safe']:
        print(f"\n  Missing keys: {results['safe']['missing_keys_pct']:.1f}%")
    
    print("=" * 60)


def export_json(results, filepath, n_rows, mode):
    """
    Export results to JSON file.
    
    Matches structure from benchmark_performance.py for compatibility.
    """
    # Create directory if needed
    dirpath = os.path.dirname(filepath)
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)
    
    # Calculate total time
    total_time = sum(results[s]['time_s'] for s in ['simple', 'safe', 'direct'])
    
    # Build metrics dict for compatibility with baseline_utils
    metrics = {
        'simple_time_s': results['simple']['time_s'],
        'simple_rows_per_sec': results['simple']['rows_per_sec'],
        'simple_peak_mb': results['simple']['peak_mb'],
        'simple_n_aliases': results['simple']['n_aliases'],
        
        'safe_time_s': results['safe']['time_s'],
        'safe_rows_per_sec': results['safe']['rows_per_sec'],
        'safe_peak_mb': results['safe']['peak_mb'],
        'safe_n_aliases': results['safe']['n_aliases'],
        'safe_missing_pct': results['safe'].get('missing_keys_pct', 0),
        
        'direct_time_s': results['direct']['time_s'],
        'direct_rows_per_sec': results['direct']['rows_per_sec'],
        'direct_peak_mb': results['direct']['peak_mb'],
        'direct_n_aliases': results['direct']['n_aliases'],
        
        'direct_vs_safe_speedup': results.get('direct_vs_safe_speedup'),
        'safe_vs_simple_ratio': results.get('safe_vs_simple_ratio'),
        'direct_vs_simple_ratio': results.get('direct_vs_simple_ratio'),
    }
    
    output = {
        'benchmark': 'benchmark_materialize_aliases.py',
        'timestamp': datetime.now().isoformat(),
        'hostname': platform.node(),
        'python_version': platform.python_version(),
        'platform': platform.platform(),
        'rows': n_rows,
        'mode': mode,
        'time_s': total_time,
        'metrics': metrics,
        'results': {
            'simple': results['simple'],
            'safe': results['safe'],
            'direct': results['direct'],
        },
    }
    
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nResults exported to: {filepath}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark materialize_aliases() performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python benchmark_materialize_aliases.py                  # Full benchmark
    python benchmark_materialize_aliases.py --quick          # Quick mode (CI)
    python benchmark_materialize_aliases.py --json out.json  # Export to JSON
    python benchmark_materialize_aliases.py --rows 1000000   # Custom size

Scenarios:
    simple:  Alias chain without subframes (baseline)
    safe:    Subframe joins with fill_mode='safe'
    direct:  Subframe joins with fill_mode='direct'
        """
    )
    parser.add_argument('--quick', action='store_true',
                        help=f'Quick mode: {QUICK_ROWS:,} rows')
    parser.add_argument('--rows', type=int, default=None,
                        help='Custom row count (overrides --quick)')
    parser.add_argument('--json', type=str, metavar='FILE',
                        help='Export results to JSON file')
    parser.add_argument('--quiet', action='store_true',
                        help='Minimal output')
    
    args = parser.parse_args()
    
    # Determine row count
    if args.rows:
        n_rows = args.rows
        mode = 'custom'
    elif args.quick:
        n_rows = QUICK_ROWS
        mode = 'quick'
    else:
        n_rows = DEFAULT_ROWS
        mode = 'full'
    
    verbose = not args.quiet
    
    # Run benchmarks
    results = run_all_benchmarks(n_rows, verbose)
    
    # Print summary
    if verbose:
        print_summary(results, n_rows)
    else:
        # Minimal output for --quiet
        total_time = sum(results[s]['time_s'] for s in ['simple', 'safe', 'direct'])
        speedup = results.get('direct_vs_safe_speedup', 0)
        print(f"Total: {total_time:.2f}s | direct_vs_safe: {speedup:.2f}x | rows: {n_rows:,}")
    
    # Export to JSON if requested
    if args.json:
        export_json(results, args.json, n_rows, mode)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
