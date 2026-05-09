#!/usr/bin/env python
"""Phase 13.22 — Calibrate K values using sum-of-primitives T_expected.

Runs each roofline test N times, computes K = T_obs / T_expected,
writes K values + T_expected + T_observed to JSON.

Usage:
    NUMBA_THREADING_LAYER=omp python benchmarks/scripts/calibrate_roofline_K.py
    NUMBA_THREADING_LAYER=omp python benchmarks/scripts/calibrate_roofline_K.py --n-runs 10
"""
import argparse
import cProfile
import io
import json
import os
import pstats
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ---- Fixture params (v1.10 §4.4) ----
SW2D_NX, SW2D_NY = 32, 32
SW2D_N_ROWS = 100_000                   # §4.4: pinned to 100K
SW2D_N_GROUPS = SW2D_NX * SW2D_NY
SW2D_RPG = SW2D_N_ROWS // SW2D_N_GROUPS  # ~97
SW2D_N_LINEAR = 2
SW2D_N_FEATURES = SW2D_N_LINEAR + 1
SW2D_WINDOW_R = 2
SW2D_N_OUTPUT_COLS = 14

S2_N_ROWS = 100_000
S2_N_GROUPS = 1000
S2_RPG = 100
S2_N_FEATURES = 3
S2_N_COLS_KEPT = 4


def _median_mad(values):
    arr = np.array(values)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    return med, mad


def _warmup(fn, max_warmup=5, threshold=0.05):
    times = []
    for i in range(max_warmup):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
        if len(times) >= 2 and times[-2] > 0:
            if abs(times[-1] - times[-2]) / times[-2] < threshold:
                return i + 1
    return max_warmup


def _count_neighbors(nx, ny, wr):
    total = 0
    for ix in range(nx):
        for iy in range(ny):
            c = 0
            for dx in range(-wr, wr+1):
                for dy in range(-wr, wr+1):
                    if 0 <= ix+dx < nx and 0 <= iy+dy < ny:
                        c += 1
            total += c
    return total / (nx * ny)


def _cprofile_cumtime(fn, target):
    pr = cProfile.Profile()
    pr.enable()
    fn()
    pr.disable()
    stats = pstats.Stats(pr, stream=io.StringIO())
    for key, (cc, nc, tt, ct, callers) in stats.stats.items():
        if target in str(key[2]):
            return ct
    return 0.0


def load_primitives(machine):
    for base in [Path("bench_out"), Path("benchmarks/bench_out")]:
        p = base / f"primitives_{machine}.json"
        if p.exists():
            with open(p) as f:
                return json.load(f)["primitives"]
    raise FileNotFoundError(f"No primitives_{machine}.json in bench_out/")


def T_expected_sw_fit(P):
    M1 = P["M1_stream_read_GBps"]["value"]
    M2 = P["M2_gather_GBps"]["value"]
    M3w = P["M3w_scatter_write_GBps"]["value"]
    C1 = P.get("C1_XtWX_ns_sw", P.get("C1_XtWX_ns_v4", P.get("C1_XtWX_ns", {})))
    C1_val = C1.get("value", 1000)
    if "C1_XtWX_ns_sw" not in P and "C1_XtWX_ns_v4" in P:
        C1_val = P["C1_XtWX_ns_v4"]["value"] * 2200 / 100
    C2 = P["C2_solve_ns"]["value"]
    n_rows = SW2D_N_ROWS
    n_groups = SW2D_N_GROUPS
    rpg = SW2D_RPG
    n_nbr = _count_neighbors(SW2D_NX, SW2D_NY, SW2D_WINDOW_R)

    T = (
        4 * n_rows * 8 / (M1 * 1e9)
        + 8 * n_rows * 8 / (M1 * 1e9)
        + 5 * n_rows * 8 / (M3w * 1e9)
        + 2 * n_rows * 8 / (M1 * 1e9)
        + 2 * n_rows * 8 / (M3w * 1e9)
        + n_groups * 8 / (M3w * 1e9)
        + n_groups * 16 / (M1 * 1e9)
        + n_groups * n_nbr * rpg * 8 / (M2 * 1e9)
        + n_groups * (C1_val + C2) * 1e-9
        + n_rows * 8 / (M3w * 1e9)
        + n_groups * SW2D_N_OUTPUT_COLS * 8 / (M3w * 1e9)
    )
    return T


def T_expected_v4(P):
    M1 = P["M1_stream_read_GBps"]["value"]
    M2 = P["M2_gather_GBps"]["value"]
    M3w = P["M3w_scatter_write_GBps"]["value"]
    C1 = P.get("C1_XtWX_ns_v4", P.get("C1_XtWX_ns", {})).get("value", 1000)
    C2 = P["C2_solve_ns"]["value"]
    C6 = P["C6_MAD_ns"]["value"]
    n_rows = S2_N_ROWS
    n_groups = S2_N_GROUPS
    n_cols = S2_N_COLS_KEPT

    T = (
        4 * n_rows * 8 / (M1 * 1e9)
        + n_groups * (C1 + C2 + C6) * 1e-9
        + n_rows * 8 / (M3w * 1e9)
        + n_rows * n_cols * 8 / (M2 * 1e9)
        + n_rows * n_cols * 8 / (M3w * 1e9)
        + n_groups * 15 * 8 / (M3w * 1e9)
    )
    return T


def T_expected_assign_bin(P):
    M1 = P["M1_stream_read_GBps"]["value"]
    M3w = P["M3w_scatter_write_GBps"]["value"]
    return 8 * SW2D_N_ROWS * 8 / (M1 * 1e9) + 5 * SW2D_N_ROWS * 8 / (M3w * 1e9)


def T_expected_csort(P):
    M1 = P["M1_stream_read_GBps"]["value"]
    M3w = P["M3w_scatter_write_GBps"]["value"]
    n_rows = SW2D_N_ROWS
    n_g = SW2D_N_GROUPS
    return (n_rows*8/(M1*1e9) + n_g*8/(M3w*1e9) + n_g*16/(M1*1e9)
            + 2*n_rows*8/(M1*1e9) + 2*n_rows*8/(M3w*1e9))


def T_expected_kernel(P):
    C1 = P.get("C1_XtWX_ns_sw", P.get("C1_XtWX_ns_v4", P.get("C1_XtWX_ns", {})))
    C1_val = C1.get("value", 1000)
    if "C1_XtWX_ns_sw" not in P and "C1_XtWX_ns_v4" in P:
        C1_val = P["C1_XtWX_ns_v4"]["value"] * 2200 / 100
    C2 = P["C2_solve_ns"]["value"]
    return SW2D_N_GROUPS * (C1_val + C2) * 1e-9


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    hostname = socket.gethostname().split('.')[0]
    n_runs = args.n_runs
    print(f"Calibrating K (sum-of-primitives) on {hostname}, {n_runs} runs")

    P = load_primitives(hostname)

    from groupby_regression_sliding_window import make_sliding_window_fit as swf
    from groupby_regression_optimized import make_parallel_fit_v4 as v4

    # Build fixtures
    rng = np.random.default_rng(13_22)
    sw_rows = []
    row_count = 0
    for ix in range(SW2D_NX):
        for iy in range(SW2D_NY):
            bin_idx = ix * SW2D_NY + iy
            n_in_bin = SW2D_RPG if bin_idx < SW2D_N_GROUPS - 1 else SW2D_N_ROWS - row_count
            for _ in range(n_in_bin):
                x1, x2 = rng.normal(), rng.normal()
                sw_rows.append({"bin_x": ix, "bin_y": iy, "x1": x1, "x2": x2,
                                "y": 2 + 0.5*x1 - 0.3*x2 + rng.normal(0, 0.1)})
            row_count += n_in_bin
    df_sw = pd.DataFrame(sw_rows)
    df_sw["bin_x"] = df_sw["bin_x"].astype(np.int64)
    df_sw["bin_y"] = df_sw["bin_y"].astype(np.int64)

    rng2 = np.random.default_rng(42)
    n = S2_N_ROWS
    df_s2 = pd.DataFrame({"group": rng2.integers(0, S2_N_GROUPS, size=n),
                           "x1": rng2.normal(size=n), "x2": rng2.normal(size=n)})
    df_s2["y"] = 2 + 0.5*df_s2["x1"] - 0.3*df_s2["x2"] + rng2.normal(0, 0.1, size=n)
    df_s2["group"] = df_s2["group"].astype(np.int64)

    call_sw = lambda: swf(df=df_sw, gb_columns=["bin_x","bin_y"], fit_columns=["y"],
                          linear_columns=["x1","x2"],
                          window_spec={"bin_x": SW2D_WINDOW_R, "bin_y": SW2D_WINDOW_R},
                          algorithm='recompute', backend='numba', min_stat=5, suffix='_sw')
    call_v4 = lambda: v4(df=df_s2, gb_columns=["group"], fit_columns=["y"],
                         linear_columns=["x1","x2"], min_stat=5, suffix='_v4',
                         backend='pyarrow')

    print("  Warming up...")
    _warmup(call_sw)
    _warmup(call_v4)

    n_nbr = _count_neighbors(SW2D_NX, SW2D_NY, SW2D_WINDOW_R)
    print(f"  SW2D: n_neighbors_eff={n_nbr:.1f}, rows_per_window={n_nbr*SW2D_RPG:.0f}")

    # T_expected values
    te_sw = T_expected_sw_fit(P)
    te_v4 = T_expected_v4(P)
    te_ab = T_expected_assign_bin(P)
    te_cs = T_expected_csort(P)
    te_fk = T_expected_kernel(P)
    print(f"  T_expected: SW={te_sw*1e3:.2f}ms V4={te_v4*1e3:.2f}ms AB={te_ab*1e3:.3f}ms CS={te_cs*1e3:.3f}ms FK={te_fk*1e3:.2f}ms")

    tests = [
        ("test_fit_regression_roofline", call_sw, te_sw, "perf_counter", None),
        ("test_v4_modeled_roofline", call_v4, te_v4, "perf_counter", None),
        ("test_assign_bin_ids_modeled_roofline", call_sw, te_ab, "cprofile", "_assign_bin_ids_fast"),
        ("test_counting_sort_modeled_roofline", call_sw, te_cs, "cprofile", "_counting_sort_indices"),
        ("test_fit_kernel_modeled_roofline", call_sw, te_fk, "perf_counter", None),
    ]

    results = {}
    for name, call_fn, t_exp, instrument, cprof_target in tests:
        print(f"\n  {name}:")
        k_vals, t_obs_vals = [], []
        for run in range(n_runs):
            if instrument == "cprofile":
                t_obs = _cprofile_cumtime(call_fn, cprof_target)
            else:
                t0 = time.perf_counter()
                call_fn()
                t_obs = time.perf_counter() - t0

            k = t_obs / t_exp if t_exp > 0 else 0
            k_vals.append(k)
            t_obs_vals.append(t_obs)
            print(f"    run {run+1}/{n_runs}: K={k:.2f} T_obs={t_obs*1e3:.1f}ms T_exp={t_exp*1e3:.2f}ms")

        k_med, k_mad = _median_mad(k_vals)
        t_med, _ = _median_mad(t_obs_vals)
        results[name] = {
            "K_median": round(k_med, 2),
            "K_mad": round(k_mad, 2),
            "n_runs": n_runs,
            "T_expected_ms": round(t_exp * 1e3, 3),
            "T_observed_ms": round(t_med * 1e3, 1),
            "K_values": [round(v, 2) for v in k_vals],
        }
        print(f"    → K_median={k_med:.2f} ± MAD={k_mad:.2f}")

    # Dispatch count (separate)
    print(f"\n  test_v4_median_dispatch_count:")
    call_batch = lambda: v4(df=df_s2, gb_columns=["group"], fit_columns=["y"],
                            linear_columns=["x1","x2"], min_stat=5, suffix='_v4',
                            backend='pyarrow')
    _warmup(call_batch)
    t0 = time.perf_counter()
    call_batch()
    t_batch = time.perf_counter() - t0

    os.environ["GBAI_DISABLE_BATCH_MEDIAN"] = "1"
    call_fb = lambda: v4(df=df_s2, gb_columns=["group"], fit_columns=["y"],
                         linear_columns=["x1","x2"], min_stat=5, suffix='_v4_fb',
                         backend='pyarrow')
    _warmup(call_fb)
    t0 = time.perf_counter()
    call_fb()
    t_fb = time.perf_counter() - t0
    os.environ.pop("GBAI_DISABLE_BATCH_MEDIAN", None)

    ratio = t_fb / t_batch if t_batch > 0 else 0
    results["test_v4_median_dispatch_count"] = {
        "t_batch_ms": round(t_batch*1e3, 1),
        "t_fallback_ms": round(t_fb*1e3, 1),
        "speedup_ratio": round(ratio, 1),
    }
    print(f"    t_batch={t_batch*1e3:.1f}ms t_fb={t_fb*1e3:.1f}ms ratio={ratio:.1f}x")

    # Threading config per v1.12 — use 'NOT_SET' sentinel
    threading_layer = os.environ.get('NUMBA_THREADING_LAYER', 'NOT_SET')
    try:
        import numba
        num_threads = numba.get_num_threads()
    except (ImportError, AttributeError):
        num_threads = None

    output = {
        "schema": "roofline_K_v2",
        "machine": hostname,
        "timestamp": datetime.now().isoformat(),
        "numba_threading_layer": threading_layer,
        "numba_num_threads": num_threads,
        "n_runs": n_runs,
        "fixture_SW2D": {"n_rows": SW2D_N_ROWS, "n_groups": SW2D_N_GROUPS,
                         "n_neighbors_eff": round(n_nbr, 1),
                         "rows_per_window": round(n_nbr * SW2D_RPG)},
        "fixture_S2": {"n_rows": S2_N_ROWS, "n_groups": S2_N_GROUPS},
        "test_K_measured": results,
    }

    if args.output:
        out_path = Path(args.output)
    else:
        out_dir = Path("bench_out")
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"roofline_K_{hostname}.json"

    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")
    print(f"\nNext steps:")
    print(f"  1. python benchmarks/scripts/update_roofline_baseline.py \\")
    print(f"       --primitives bench_out/primitives_{hostname}.json \\")
    print(f"       --k-values {out_path} \\")
    print(f"       --output benchmarks/baselines/roofline_baseline.json")
    print(f"  2. NUMBA_THREADING_LAYER=omp pytest tests/test_performance_roofline.py -v -s -m roofline")


if __name__ == "__main__":
    main()
