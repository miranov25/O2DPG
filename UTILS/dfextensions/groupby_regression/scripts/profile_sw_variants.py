#!/usr/bin/env python3
"""Profile sliding window variants (V1/V2/V3/V3-Numba) at TPC-like scale.

Usage:
    python scripts/profile_sw_variants.py [--grid 15] [--entries 10] [--window 1] [--cprofile]

Outputs timing breakdown for each variant and identifies bottlenecks.
"""
import argparse
import itertools
import sys
import time

import numpy as np
import pandas as pd


def make_grid(n_bins: int, entries_per_bin: int):
    """Dense cubic grid with known linear truth: y = 2x + 1 + noise."""
    rng = np.random.default_rng(42)
    bins = np.array(list(itertools.product(range(n_bins), repeat=3)))
    bins_expanded = np.repeat(bins, entries_per_bin, axis=0)
    df = pd.DataFrame(bins_expanded, columns=['xBin', 'yBin', 'zBin']).astype(np.int32)
    df['x'] = rng.normal(0.0, 1.0, len(df))
    df['value'] = 2.0 * df['x'] + 1.0 + 0.1 * rng.normal(0.0, 1.0, len(df))
    return df


def bench(df, label, window, algorithm, backend, n_warmup=1, n_runs=3):
    """Benchmark with warmup and multiple runs."""
    try:
        from groupby_regression_sliding_window import make_sliding_window_fit
    except ImportError:
        from dfextensions.groupby_regression.groupby_regression_sliding_window import make_sliding_window_fit

    ws = {'xBin': window, 'yBin': window, 'zBin': window}
    kw = dict(
        df=df, gb_columns=['xBin', 'yBin', 'zBin'],
        window_spec=ws, fit_columns=['value'], linear_columns=['x'],
        min_stat=5, algorithm=algorithm, backend=backend, suffix='',
    )

    # Warmup (JIT compilation)
    for _ in range(n_warmup):
        make_sliding_window_fit(**kw)

    # Timed runs
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        r = make_sliding_window_fit(**kw, return_metadata=True)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

    result_df, meta = r
    backend_used = meta.get('backend_used', '?')

    t_min = min(times)
    t_med = sorted(times)[len(times) // 2]
    print(f"  {label:30s}  min={t_min:.3f}s  med={t_med:.3f}s  backend={backend_used}")
    return t_med


def profile_cprofile(df, window, algorithm, backend, label):
    """Run cProfile on a single variant."""
    import cProfile
    import pstats
    from io import StringIO
    try:
        from groupby_regression_sliding_window import make_sliding_window_fit
    except ImportError:
        from dfextensions.groupby_regression.groupby_regression_sliding_window import make_sliding_window_fit

    ws = {'xBin': window, 'yBin': window, 'zBin': window}

    # Warmup
    make_sliding_window_fit(
        df=df, gb_columns=['xBin', 'yBin', 'zBin'],
        window_spec=ws, fit_columns=['value'], linear_columns=['x'],
        min_stat=5, algorithm=algorithm, backend=backend, suffix='',
    )

    # Profile
    pr = cProfile.Profile()
    pr.enable()
    make_sliding_window_fit(
        df=df, gb_columns=['xBin', 'yBin', 'zBin'],
        window_spec=ws, fit_columns=['value'], linear_columns=['x'],
        min_stat=5, algorithm=algorithm, backend=backend, suffix='',
    )
    pr.disable()

    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(30)
    print(f"\n=== cProfile: {label} ===")
    print(s.getvalue())


def main():
    parser = argparse.ArgumentParser(description="Profile SW variants")
    parser.add_argument('--grid', type=int, default=15, help='Bins per dim (default: 15)')
    parser.add_argument('--entries', type=int, default=10, help='Entries per bin (default: 10)')
    parser.add_argument('--window', type=int, default=1, help='Window half-width (default: 1)')
    parser.add_argument('--cprofile', action='store_true', help='Run cProfile on each variant')
    args = parser.parse_args()

    df = make_grid(args.grid, args.entries)
    n_bins = args.grid ** 3
    n_rows = len(df)
    n_nbr = (2 * args.window + 1) ** 3
    print(f"Grid: {args.grid}³ = {n_bins} bins, {n_rows} rows, window=±{args.window} ({n_nbr} neighbors)")

    # Check Numba availability
    try:
        import numba
        has_numba = True
        print(f"Numba: {numba.__version__}")
    except ImportError:
        has_numba = False
        print("Numba: not available")

    print(f"\n{'='*70}")
    print("Timing (3 runs after warmup):")
    print(f"{'='*70}")

    results = {}
    results['V1-numpy'] = bench(df, 'V1-numpy (recompute+lstsq)',
                                 args.window, 'recompute', 'numpy')

    if has_numba:
        results['V2-numba'] = bench(df, 'V2-numba (recompute+kernel)',
                                     args.window, 'recompute', 'numba')

    results['V3-numpy'] = bench(df, 'V3-numpy (incremental+solve)',
                                 args.window, 'incremental', 'numpy')

    if has_numba:
        results['V3-numba'] = bench(df, 'V3-numba (incremental+cholesky)',
                                     args.window, 'incremental', 'numba')

    print(f"\n{'='*70}")
    print("Relative performance:")
    print(f"{'='*70}")
    baseline = results['V1-numpy']
    for label, t in results.items():
        ratio = baseline / t if t > 0 else float('inf')
        print(f"  {label:30s}  {ratio:.1f}× vs V1-numpy")

    if has_numba and 'V2-numba' in results and 'V3-numba' in results:
        r = results['V2-numba'] / results['V3-numba']
        print(f"  {'V3-numba vs V2-numba':30s}  {r:.1f}×")

    if args.cprofile:
        profile_cprofile(df, args.window, 'recompute', 'numpy', 'V1-numpy')
        if has_numba:
            profile_cprofile(df, args.window, 'recompute', 'numba', 'V2-numba')
        profile_cprofile(df, args.window, 'incremental', 'numpy', 'V3-numpy')
        if has_numba:
            profile_cprofile(df, args.window, 'incremental', 'numba', 'V3-numba')


if __name__ == '__main__':
    main()
