#!/usr/bin/env python
"""Phase 13.22.GB-RooflineTier1 — Primitive Calibration Script (D5).

Measures phase 12.12b primitives (Appendix A, lines 933-941) on the
current machine at the working-set sizes used by Tier 1 tests.

Per v1.10 §4.5b: primitives are measured at the fixture's actual
working-set size, not a single size. C1/C2 are measured at BOTH
V4 operand size (rows_per_group=100) and SW fit operand size
(rows_per_window=2200).

Usage:
    NUMBA_THREADING_LAYER=omp python benchmarks/scripts/measure_primitives.py
"""
import argparse
import json
import os
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


def _median_mad(values):
    arr = np.array(values)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    return med, mad


# ---- M-primitives (bandwidth) ----

def measure_M1(n_rows, n_runs=20):
    arr = np.random.randn(n_rows).astype(np.float64)
    np.sum(arr)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        np.sum(arr)
        times.append(time.perf_counter() - t0)
    gbps = [n_rows * 8 / t / 1e9 for t in times]
    med, mad = _median_mad(np.array(gbps))
    return {"value": round(med, 2), "mad": round(mad, 3), "unit": "GB/s",
            "n_runs": n_runs, "n_rows": n_rows, "source_line": "phase_12_12b A:934"}


def measure_M2(n_rows, n_runs=20):
    data = np.random.randn(n_rows).astype(np.float64)
    idx = np.random.randint(0, n_rows, size=n_rows, dtype=np.int64)
    out = np.empty(n_rows, dtype=np.float64)
    out[:] = data[idx]
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        out[:] = data[idx]
        times.append(time.perf_counter() - t0)
    gbps = [n_rows * 8 * 2 / t / 1e9 for t in times]
    med, mad = _median_mad(np.array(gbps))
    return {"value": round(med, 2), "mad": round(mad, 3), "unit": "GB/s",
            "n_runs": n_runs, "n_rows": n_rows, "source_line": "phase_12_12b A:935"}


def measure_M3w(n_rows, n_runs=20):
    data = np.random.randn(n_rows).astype(np.float64)
    idx = np.random.randint(0, n_rows, size=n_rows, dtype=np.int64)
    out = np.empty(n_rows, dtype=np.float64)
    out[idx] = data
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        out[idx] = data
        times.append(time.perf_counter() - t0)
    gbps = [n_rows * 8 * 2 / t / 1e9 for t in times]
    med, mad = _median_mad(np.array(gbps))
    return {"value": round(med, 2), "mad": round(mad, 3), "unit": "GB/s",
            "n_runs": n_runs, "n_rows": n_rows, "source_line": "phase_12_12b A:937"}


# ---- C-primitives (compute, operand-size-aware per v1.10 §4.2) ----

def measure_C1(n_groups, rows_per_call, n_features, n_runs=20):
    """C1: XtWX construction at specified operand size."""
    n_params = n_features
    rng = np.random.default_rng(42)
    Xs = [rng.standard_normal((rows_per_call, n_params)).astype(np.float64)
          for _ in range(min(n_groups, 100))]
    for X in Xs[:3]:
        _ = X.T @ X
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        for i in range(n_groups):
            _ = Xs[i % len(Xs)].T @ Xs[i % len(Xs)]
        times.append(time.perf_counter() - t0)
    ns_per = [t / n_groups * 1e9 for t in times]
    med, mad = _median_mad(np.array(ns_per))
    return {"value": round(med, 1), "mad": round(mad, 1), "unit": "ns/call",
            "n_runs": n_runs, "n_groups": n_groups,
            "rows_per_call": rows_per_call, "n_features": n_features,
            "source_line": "phase_12_12b A:938"}


def measure_C2(n_groups, n_features, n_runs=20):
    """C2: Small Cholesky solve."""
    n_params = n_features
    rng = np.random.default_rng(42)
    As, bs = [], []
    for _ in range(min(n_groups, 100)):
        X = rng.standard_normal((50, n_params)).astype(np.float64)
        As.append(X.T @ X)
        bs.append(X.T @ rng.standard_normal(50))
    for i in range(3):
        np.linalg.solve(As[i], bs[i])
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        for i in range(n_groups):
            np.linalg.solve(As[i % len(As)], bs[i % len(bs)])
        times.append(time.perf_counter() - t0)
    ns_per = [t / n_groups * 1e9 for t in times]
    med, mad = _median_mad(np.array(ns_per))
    return {"value": round(med, 1), "mad": round(mad, 1), "unit": "ns/call",
            "n_runs": n_runs, "n_groups": n_groups, "n_features": n_features,
            "source_line": "phase_12_12b A:939"}


def measure_C6(n_groups, rows_per_call, n_runs=20):
    """C6: MAD (two medians per call)."""
    rng = np.random.default_rng(42)
    residuals = [rng.standard_normal(rows_per_call).astype(np.float64)
                 for _ in range(min(n_groups, 100))]
    for r in residuals[:3]:
        np.median(np.abs(r - np.median(r)))
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        for i in range(n_groups):
            r = residuals[i % len(residuals)]
            np.median(np.abs(r - np.median(r)))
        times.append(time.perf_counter() - t0)
    ns_per = [t / n_groups * 1e9 for t in times]
    med, mad = _median_mad(np.array(ns_per))
    return {"value": round(med, 1), "mad": round(mad, 1), "unit": "ns/call",
            "n_runs": n_runs, "n_groups": n_groups,
            "rows_per_call": rows_per_call, "source_line": "phase_12_12b A:940"}


def main():
    parser = argparse.ArgumentParser(description="Measure phase 12.12b primitives")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--n-rows", type=int, default=100_000)
    parser.add_argument("--n-runs", type=int, default=20)
    args = parser.parse_args()

    hostname = socket.gethostname().split('.')[0]
    n_rows = args.n_rows
    n_runs = args.n_runs

    print(f"Measuring primitives on {hostname} (n_rows={n_rows}, n_runs={n_runs})")
    print(f"  Reference: phase_12_12b_specification_v2.md Appendix A, lines 933-941")
    print()

    results = {
        "schema": "primitives_v1.1",
        "machine": hostname,
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
    }
    try:
        import numba
        results["numba_version"] = numba.__version__
        results["numba_threading_layer"] = os.environ.get("NUMBA_THREADING_LAYER", "NOT_SET")
        try:
            results["numba_num_threads"] = numba.get_num_threads()
        except AttributeError:
            results["numba_num_threads"] = numba.config.NUMBA_NUM_THREADS
    except ImportError:
        results["numba_version"] = "not installed"

    primitives = {}

    # M-primitives at fixture working-set size
    for name, func, kw in [
        ("M1_stream_read_GBps", measure_M1, {"n_rows": n_rows, "n_runs": n_runs}),
        ("M2_gather_GBps", measure_M2, {"n_rows": n_rows, "n_runs": n_runs}),
        ("M3w_scatter_write_GBps", measure_M3w, {"n_rows": n_rows, "n_runs": n_runs}),
    ]:
        print(f"  {name}...", end=" ", flush=True)
        r = func(**kw)
        primitives[name] = r
        print(f"{r['value']} ± {r['mad']} {r['unit']}")

    # C-primitives at V4 operand size (rows_per_group=100, n_features=3)
    for name, func, kw in [
        ("C1_XtWX_ns_v4", measure_C1, {"n_groups": 1000, "rows_per_call": 100, "n_features": 3, "n_runs": n_runs}),
        ("C2_solve_ns", measure_C2, {"n_groups": 1000, "n_features": 3, "n_runs": n_runs}),
        ("C6_MAD_ns", measure_C6, {"n_groups": 1000, "rows_per_call": 100, "n_runs": n_runs}),
    ]:
        print(f"  {name}...", end=" ", flush=True)
        r = func(**kw)
        primitives[name] = r
        print(f"{r['value']} ± {r['mad']} {r['unit']}")

    # C1 at SW fit operand size (rows_per_window=2200, n_features=3)
    print(f"  C1_XtWX_ns_sw...", end=" ", flush=True)
    r = measure_C1(n_groups=100, rows_per_call=2200, n_features=3, n_runs=n_runs)
    primitives["C1_XtWX_ns_sw"] = r
    print(f"{r['value']} ± {r['mad']} {r['unit']}")

    results["primitives"] = primitives

    if args.output:
        out_path = Path(args.output)
    else:
        out_dir = Path("bench_out")
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"primitives_{hostname}.json"

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
