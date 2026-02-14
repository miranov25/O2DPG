"""
Parallel sliding window benchmark.

Measures:
  A. Serial baseline: flatten + per-unit V5arr times
  B. Fork overhead: empty worker dispatch cost
  C. Data copy: pickle/unpickle cost per unit
  D. Parallel scaling: 1-36 workers with verbose timing
  E. Sort algorithm comparison: argsort vs counting sort
  F. Validation: serial vs parallel coefficient agreement

Usage:
    python bench_slidingwindow_parallel.py                # full benchmark
    python bench_slidingwindow_parallel.py --cprofile     # + cProfile
"""
import sys, time, os, multiprocessing
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ.setdefault('NUMBA_NUM_THREADS', '1')

from groupby_regression_sliding_window import (
    make_sliding_window_fit,
    make_sliding_window_fit_parallel,
    make_sliding_window_fit_v5_arrays,
    _flatten_bins_for_v5,
    _counting_sort_indices,
)

DO_CPROFILE = '--cprofile' in sys.argv
PROFILE_DIR = 'benchmark_results/profiles'

GB = ['xBin', 'yBin', 'zBin']
FIT = ['value']
LIN = ['x']
WS = {'xBin': 1, 'yBin': 1, 'zBin': 0}
SPLIT = ['sector']


def make_data(n_sectors, grid, rpb, seed=42):
    rng = np.random.RandomState(seed)
    frames = []
    for sec in range(n_sectors):
        coords = np.array(np.meshgrid(*[np.arange(grid)] * 3)).T.reshape(-1, 3)
        coords_rep = np.repeat(coords, rpb, axis=0)
        n = len(coords_rep)
        x = rng.standard_normal(n)
        value = (2.0 + 0.1 * sec) * x + 1.0 + rng.standard_normal(n) * 0.1
        frames.append(pd.DataFrame({
            'sector': sec, 'xBin': coords_rep[:, 0],
            'yBin': coords_rep[:, 1], 'zBin': coords_rep[:, 2],
            'x': x, 'value': value,
        }))
    df = pd.concat(frames, ignore_index=True)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def noop_worker(x):
    """Empty worker -- measures fork + dispatch overhead."""
    return len(x)


def pickle_roundtrip(arrays):
    """Measure pickle cost of numpy arrays."""
    import pickle
    t0 = time.perf_counter()
    data = pickle.dumps(arrays)
    t_dump = time.perf_counter() - t0
    t0 = time.perf_counter()
    pickle.loads(data)
    t_load = time.perf_counter() - t0
    return t_dump, t_load, len(data)


def main():
    grid = 25
    rpb = 200
    n_sec = 36

    print("=" * 70)
    print(f"PARALLEL SLIDING WINDOW BENCHMARK")
    print(f"Config: {n_sec} sectors, grid={grid}, rpb={rpb}")
    print("=" * 70)

    if DO_CPROFILE:
        os.makedirs(PROFILE_DIR, exist_ok=True)
        print(f"cProfile enabled -- output to {PROFILE_DIR}/")

    df = make_data(n_sec, grid, rpb)
    n_rows = len(df)
    print(f"Total rows: {n_rows:,d}")

    # ---- A. Serial per-sector breakdown ----
    print(f"\n{'='*70}")
    print("A. SERIAL PER-SECTOR BREAKDOWN")
    print(f"{'='*70}")

    df_sec0 = df[df['sector'] == 0].copy()
    n_sec_rows = len(df_sec0)
    print(f"  Sector 0: {n_sec_rows:,d} rows")

    # Warmup
    bin_ids, X, Y, n_bins, bin_coords, bounds = \
        _flatten_bins_for_v5(df_sec0, GB, FIT, LIN, None, True)
    full_ws = {d: WS.get(d, 0) for d in GB}
    make_sliding_window_fit_v5_arrays(
        bin_ids=bin_ids, X_all=X, Y_all=Y, n_bins=n_bins,
        bin_coords=bin_coords, bounds=bounds,
        gb_columns=GB, fit_columns=FIT, linear_columns=LIN,
        window_spec=full_ws, min_stat=5)

    # Timed
    times_flat = []
    times_v5 = []
    for _ in range(3):
        t0 = time.perf_counter()
        bin_ids, X, Y, n_bins, bin_coords, bounds = \
            _flatten_bins_for_v5(df_sec0, GB, FIT, LIN, None, True)
        times_flat.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        make_sliding_window_fit_v5_arrays(
            bin_ids=bin_ids, X_all=X, Y_all=Y, n_bins=n_bins,
            bin_coords=bin_coords, bounds=bounds,
            gb_columns=GB, fit_columns=FIT, linear_columns=LIN,
            window_spec=full_ws, min_stat=5)
        times_v5.append(time.perf_counter() - t0)

    tf = min(times_flat)
    tv = min(times_v5)
    print(f"  Flatten:   {tf*1e3:7.1f}ms")
    print(f"  V5arr:     {tv*1e3:7.1f}ms")
    print(f"  Total:     {(tf+tv)*1e3:7.1f}ms")
    print(f"  x36 serial: {(tf+tv)*36:.2f}s")

    # ---- B. Fork overhead ----
    print(f"\n{'='*70}")
    print("B. FORK / DISPATCH OVERHEAD")
    print(f"{'='*70}")

    from concurrent.futures import ProcessPoolExecutor, as_completed

    dummy = np.zeros(100)
    for nw in [1, 4, 8, 36]:
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=nw) as ex:
            futs = [ex.submit(noop_worker, dummy) for _ in range(36)]
            for f in as_completed(futs):
                f.result()
        t = time.perf_counter() - t0
        print(f"  {nw:3d} workers, 36 tasks: {t*1e3:7.1f}ms ({t/36*1e3:.1f}ms/task)")

    # ---- C. Pickle / data transfer cost ----
    print(f"\n{'='*70}")
    print("C. PICKLE COST (per-sector arrays)")
    print(f"{'='*70}")

    sector_mask = df['sector'].values == 0
    gb_unit = {c: df[c].values[sector_mask].copy() for c in GB}
    x_unit = df['x'].values[sector_mask].reshape(-1, 1).copy()
    y_unit = df['value'].values[sector_mask].reshape(-1, 1).copy()
    payload = (gb_unit, x_unit, y_unit)

    t_dump, t_load, nbytes = pickle_roundtrip(payload)
    print(f"  Rows: {sector_mask.sum():,d}")
    print(f"  Pickle size: {nbytes/1e6:.1f} MB")
    print(f"  Dump:  {t_dump*1e3:.1f}ms")
    print(f"  Load:  {t_load*1e3:.1f}ms")
    print(f"  Total: {(t_dump+t_load)*1e3:.1f}ms")
    print(f"  x36 sectors: {(t_dump+t_load)*36:.2f}s")

    # ---- D. Parallel with 5-point timing from function ----
    print(f"\n{'='*70}")
    print("D. PARALLEL SCALING (verbose=1)")
    print(f"{'='*70}")

    import logging
    logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)

    # Warmup
    make_sliding_window_fit_parallel(
        df=df, gb_columns=GB, fit_columns=FIT, linear_columns=LIN,
        split_columns=SPLIT, n_workers=1, window_spec=WS,
        min_stat=5, suffix='_sw')

    print(f"\n  {'Workers':>8s} | {'Total':>8s} {'Speedup':>8s}")
    print(f"  {'-'*8}-+-{'-'*8}-{'-'*8}")

    t_base = None
    parallel_results = {}
    for nw in [1, 4, 8, 16, 36]:
        t0 = time.perf_counter()
        r = make_sliding_window_fit_parallel(
            df=df, gb_columns=GB, fit_columns=FIT, linear_columns=LIN,
            split_columns=SPLIT, n_workers=nw, window_spec=WS,
            min_stat=5, suffix='_sw', verbose=1)
        t = time.perf_counter() - t0
        if t_base is None:
            t_base = t
        print(f"  {nw:8d} | {t:7.2f}s {t_base/t:7.2f}x")
        parallel_results[nw] = r

    # ---- E. Sort algorithm comparison ----
    print(f"\n{'='*70}")
    print("E. SORT ALGORITHM COMPARISON (112M int64 with 36 unique values)")
    print(f"{'='*70}")

    sector_col = df['sector'].to_numpy(dtype=np.int64)
    n = len(sector_col)

    t0 = time.perf_counter()
    order1 = np.argsort(sector_col, kind='mergesort')
    t_merge = time.perf_counter() - t0
    print(f"  argsort int64 mergesort: {t_merge:.3f}s")

    sid_u32 = sector_col.astype(np.uint32)
    t0 = time.perf_counter()
    order2 = np.argsort(sid_u32, kind='stable')
    t_u32 = time.perf_counter() - t0
    print(f"  argsort uint32 stable:   {t_u32:.3f}s")

    # Numba counting sort warmup
    _counting_sort_indices(np.array([0, 1, 0], dtype=np.int64), 2)
    t0 = time.perf_counter()
    order3, offsets3 = _counting_sort_indices(sector_col, 36)
    t_csort = time.perf_counter() - t0
    print(f"  counting sort (Numba):   {t_csort:.3f}s")
    print(f"  speedup vs mergesort:    {t_merge/t_csort:.1f}x")

    gb_raw = np.empty((n, 3), dtype=np.int64)
    for d, c in enumerate(GB):
        gb_raw[:, d] = df[c].to_numpy(dtype=np.int64)
    x_raw = df['x'].to_numpy(dtype=np.float64).reshape(-1, 1)
    y_raw = df['value'].to_numpy(dtype=np.float64).reshape(-1, 1)

    t0 = time.perf_counter()
    gb_sorted = gb_raw[order3]
    x_sorted = x_raw[order3]
    y_sorted = y_raw[order3]
    t_reorder = time.perf_counter() - t0
    print(f"  reorder 3 arrays:        {t_reorder:.3f}s")
    print(f"  total (csort + reorder): {t_csort + t_reorder:.3f}s")
    print(f"  total (old argsort+reorder): {t_merge + t_reorder:.3f}s")

    # ---- F. Validation: serial vs parallel ----
    print(f"\n{'='*70}")
    print("F. VALIDATION: SERIAL vs PARALLEL")
    print(f"{'='*70}")

    # Serial reference (single-threaded V5)
    r_serial = make_sliding_window_fit_parallel(
        df=df, gb_columns=GB, fit_columns=FIT, linear_columns=LIN,
        split_columns=SPLIT, n_workers=1, window_spec=WS,
        min_stat=5, suffix='_sw')

    slope_col = 'value_slope_x_sw'
    intercept_col = 'value_intercept_sw'

    for nw, r_par in parallel_results.items():
        if nw == 1:
            continue
        # Merge on gb + sector
        merge_cols = SPLIT + GB
        merged = r_serial[merge_cols + [slope_col]].merge(
            r_par[merge_cols + [slope_col]], on=merge_cols, suffixes=('_ser', '_par'))
        v_ser = merged[f'{slope_col}_ser'].values
        v_par = merged[f'{slope_col}_par'].values
        mask = np.isfinite(v_ser) & np.isfinite(v_par)
        if mask.sum() > 0:
            max_diff = np.max(np.abs(v_ser[mask] - v_par[mask]))
            ok = max_diff < 1e-10
            status = "PASS" if ok else "FAIL"
            print(f"  w={nw:2d} vs w=1: max|diff|={max_diff:.2e} ({mask.sum()} bins) {status}")
        else:
            print(f"  w={nw:2d} vs w=1: no comparable bins")

    # ---- cProfile ----
    if DO_CPROFILE:
        import cProfile, pstats, io, zipfile

        profile_files = []

        # Profile serial
        print(f"\n{'='*70}")
        print("CPROFILE: n_workers=1 (serial)")
        print(f"{'='*70}")
        pr = cProfile.Profile()
        pr.enable()
        make_sliding_window_fit_parallel(
            df=df, gb_columns=GB, fit_columns=FIT, linear_columns=LIN,
            split_columns=SPLIT, n_workers=1, window_spec=WS,
            min_stat=5, suffix='_sw', verbose=1)
        pr.disable()

        prof_file = os.path.join(PROFILE_DIR, 'parallel_w1_serial.prof')
        pr.dump_stats(prof_file)
        profile_files.append(prof_file)

        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats('tottime').print_stats(30)
        print(s.getvalue())

        # Save CSV
        csv_file = os.path.join(PROFILE_DIR, 'parallel_w1_serial.csv')
        lines = s.getvalue().split('\n')
        with open(csv_file, 'w') as f:
            f.write('ncalls,tottime,percall_tot,cumtime,percall_cum,function\n')
            in_table = False
            for line in lines:
                line = line.strip()
                if line.startswith('ncalls'):
                    in_table = True
                    continue
                if in_table and line:
                    parts = line.split(None, 5)
                    if len(parts) >= 6:
                        f.write(','.join(parts) + '\n')
        profile_files.append(csv_file)

        # Profile parallel w=4
        print(f"\n{'='*70}")
        print("CPROFILE: n_workers=4 (parallel)")
        print(f"{'='*70}")
        pr2 = cProfile.Profile()
        pr2.enable()
        make_sliding_window_fit_parallel(
            df=df, gb_columns=GB, fit_columns=FIT, linear_columns=LIN,
            split_columns=SPLIT, n_workers=4, window_spec=WS,
            min_stat=5, suffix='_sw', verbose=1)
        pr2.disable()

        prof_file2 = os.path.join(PROFILE_DIR, 'parallel_w4.prof')
        pr2.dump_stats(prof_file2)
        profile_files.append(prof_file2)

        s2 = io.StringIO()
        pstats.Stats(pr2, stream=s2).sort_stats('tottime').print_stats(30)
        print(s2.getvalue())

        csv_file2 = os.path.join(PROFILE_DIR, 'parallel_w4.csv')
        lines2 = s2.getvalue().split('\n')
        with open(csv_file2, 'w') as f:
            f.write('ncalls,tottime,percall_tot,cumtime,percall_cum,function\n')
            in_table = False
            for line in lines2:
                line = line.strip()
                if line.startswith('ncalls'):
                    in_table = True
                    continue
                if in_table and line:
                    parts = line.split(None, 5)
                    if len(parts) >= 6:
                        f.write(','.join(parts) + '\n')
        profile_files.append(csv_file2)

        # Package
        zip_name = os.path.join(PROFILE_DIR, 'profiles_parallel.zip')
        with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zf:
            for pf in profile_files:
                zf.write(pf, os.path.basename(pf))
        print(f"\nProfiles packaged: {zip_name} ({len(profile_files)} files)")

    print("\n" + "=" * 70 + "\nDONE\n" + "=" * 70)


if __name__ == '__main__':
    multiprocessing.set_start_method('fork', force=True)
    main()
