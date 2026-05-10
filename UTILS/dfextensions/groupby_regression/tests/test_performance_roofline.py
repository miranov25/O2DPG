"""Phase 13.22.GB-RooflineTier1 — CI roofline tests (v1.10 proposal).

K = T_observed / T_expected where T_expected = Σ(n_ops × t_primitive).
K = 1 means at-roofline. K = 3 means 3× above ideal compiled code.

9 pytest items: 5 K-tests + 1 dispatch + 3 meta-tests.
Primitives: phase_12_12b Appendix A, lines 933-941.
Methodology: AI_Review_Scientific_Methodology v1.0-FINAL § Hardware-Limit Performance.

Run:  NUMBA_THREADING_LAYER=omp pytest tests/test_performance_roofline.py -v -s -m roofline
"""
import json
import os
import socket
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---- Paths ----
BASELINE_PATHS = [
    Path(__file__).parent.parent / "benchmarks" / "baselines" / "roofline_baseline.json",
    Path("benchmarks/baselines/roofline_baseline.json"),
]

PRIMITIVES_PATHS = [
    Path(__file__).parent.parent / "bench_out",
    Path("bench_out"),
]


# ---- SW2D fixture params (v1.10 §4.4 Fixture B) ----
SW2D_NX, SW2D_NY = 32, 32
SW2D_N_ROWS = 100_000                   # §4.4: pinned to 100K
SW2D_N_GROUPS = SW2D_NX * SW2D_NY      # 1024
SW2D_RPG = SW2D_N_ROWS // SW2D_N_GROUPS  # ~97 rows_per_group
SW2D_N_LINEAR = 2
SW2D_N_FIT = 1
SW2D_N_FEATURES = SW2D_N_LINEAR + 1    # 3 (with intercept)
SW2D_WINDOW_R = 2
SW2D_N_OUTPUT_COLS = 14                 # source-verified v1.9

# S2/V4 fixture params (v1.10 §4.4 Fixture A)
S2_N_ROWS = 100_000
S2_N_GROUPS = 1000
S2_RPG = 100
S2_N_FEATURES = 3
S2_N_COLS_KEPT = 4                      # source-verified v1.9


# ---- Helpers ----

def _load_primitives():
    """Load primitives JSON for current machine."""
    machine = os.environ.get("GBAI_ROOFLINE_MACHINE",
                             socket.gethostname().split('.')[0])
    for base in PRIMITIVES_PATHS:
        p = base / f"primitives_{machine}.json"
        if p.exists():
            with open(p) as f:
                return json.load(f)
    # Fallback: any primitives file
    for base in PRIMITIVES_PATHS:
        if base.exists():
            for f in base.glob("primitives_*.json"):
                with open(f) as fh:
                    return json.load(fh)
    return None


def _load_baseline():
    for p in BASELINE_PATHS:
        if p.exists():
            with open(p) as f:
                return json.load(f)
    return None


def _get_threshold(baseline, test_name):
    if baseline is None:
        return None
    machine = os.environ.get("GBAI_ROOFLINE_MACHINE",
                             socket.gethostname().split('.')[0])
    t = baseline.get("thresholds", {}).get(test_name, {})
    for key in [f"K_threshold_{machine}", "K_threshold_alma2"]:
        if key in t:
            return t[key]
    return None


def _warmup_until_stable(fn, max_warmup=5, threshold=0.05):
    times = []
    for i in range(max_warmup):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
        if len(times) >= 2 and times[-2] > 0:
            if abs(times[-1] - times[-2]) / times[-2] < threshold:
                return i + 1
    return max_warmup


def _count_effective_neighbors(nx, ny, wr):
    """Count average effective neighbors for a 2D grid with window_radius."""
    total = 0
    for ix in range(nx):
        for iy in range(ny):
            count = 0
            for dx in range(-wr, wr + 1):
                for dy in range(-wr, wr + 1):
                    if 0 <= ix + dx < nx and 0 <= iy + dy < ny:
                        count += 1
            total += count
    return total / (nx * ny)


# ---- T_expected formulas (v1.10 §4.5) ----

def _T_expected_sw_fit(prims):
    """T_expected for make_sliding_window_fit per v1.10 §4.5."""
    M1 = prims["M1_stream_read_GBps"]["value"]
    M2 = prims["M2_gather_GBps"]["value"]
    M3w = prims["M3w_scatter_write_GBps"]["value"]
    # C1 at SW window size (2200 rows) if available, else scale from V4
    if "C1_XtWX_ns_sw" in prims:
        C1 = prims["C1_XtWX_ns_sw"]["value"]
    elif "C1_XtWX_ns_v4" in prims:
        # Scale: C1 ∝ rows_per_call
        C1_v4 = prims["C1_XtWX_ns_v4"]["value"]
        C1 = C1_v4 * 2200 / 100  # rough scaling
    else:
        C1 = prims.get("C1_XtWX_ns", {}).get("value", 1000)
    C2 = prims["C2_solve_ns"]["value"]

    n_rows = SW2D_N_ROWS
    n_groups = SW2D_N_GROUPS
    rpg = SW2D_RPG
    n_nbr = _count_effective_neighbors(SW2D_NX, SW2D_NY, SW2D_WINDOW_R)
    n_out = SW2D_N_OUTPUT_COLS

    T = (
        4 * n_rows * 8 / (M1 * 1e9)           # input stream
        + 8 * n_rows * 8 / (M1 * 1e9)         # bin-id reads (~8 passes)
        + 5 * n_rows * 8 / (M3w * 1e9)        # bin-id writes (~5 passes)
        + 2 * n_rows * 8 / (M1 * 1e9)         # counting-sort reads
        + 2 * n_rows * 8 / (M3w * 1e9)        # counting-sort writes
        + n_groups * 8 / (M3w * 1e9)          # bincount writes
        + n_groups * 16 / (M1 * 1e9)          # cumsum
        + n_groups * n_nbr * rpg * 8 / (M2 * 1e9)  # gather
        + n_groups * (C1 + C2) * 1e-9         # OLS (window-sized)
        + n_rows * 8 / (M3w * 1e9)            # output writes
        + n_groups * n_out * 8 / (M3w * 1e9)  # output assembly (M3w)
    )
    return T, {"n_neighbors_eff": round(n_nbr, 1), "C1_sw": C1, "C2": C2}


def _T_expected_v4(prims):
    """T_expected for make_parallel_fit_v4 per v1.10 §4.5."""
    M1 = prims["M1_stream_read_GBps"]["value"]
    M2 = prims["M2_gather_GBps"]["value"]
    M3w = prims["M3w_scatter_write_GBps"]["value"]
    C1 = prims.get("C1_XtWX_ns_v4", prims.get("C1_XtWX_ns", {})).get("value", 1000)
    C2 = prims["C2_solve_ns"]["value"]
    C6 = prims["C6_MAD_ns"]["value"]

    n_rows = S2_N_ROWS
    n_groups = S2_N_GROUPS
    n_cols = S2_N_COLS_KEPT
    n_out_v4 = 15  # approximate

    T = (
        4 * n_rows * 8 / (M1 * 1e9)
        + n_groups * (C1 + C2 + C6) * 1e-9
        + n_rows * 8 / (M3w * 1e9)
        + n_rows * n_cols * 8 / (M2 * 1e9)      # df_sorted gather
        + n_rows * n_cols * 8 / (M3w * 1e9)     # df_sorted write
        + n_groups * n_out_v4 * 8 / (M3w * 1e9) # dfGB write
    )
    return T, {"C1_v4": C1, "C2": C2, "C6": C6}


def _T_expected_assign_bin_ids(prims):
    M1 = prims["M1_stream_read_GBps"]["value"]
    M3w = prims["M3w_scatter_write_GBps"]["value"]
    n_rows = SW2D_N_ROWS
    T = 8 * n_rows * 8 / (M1 * 1e9) + 5 * n_rows * 8 / (M3w * 1e9)
    return T, {}


def _T_expected_counting_sort(prims):
    M1 = prims["M1_stream_read_GBps"]["value"]
    M3w = prims["M3w_scatter_write_GBps"]["value"]
    n_rows = SW2D_N_ROWS
    n_groups = SW2D_N_GROUPS
    T = (
        n_rows * 8 / (M1 * 1e9)
        + n_groups * 8 / (M3w * 1e9)
        + n_groups * 16 / (M1 * 1e9)
        + 2 * n_rows * 8 / (M1 * 1e9)
        + 2 * n_rows * 8 / (M3w * 1e9)
    )
    return T, {}


def _T_expected_fit_kernel(prims):
    if "C1_XtWX_ns_sw" in prims:
        C1 = prims["C1_XtWX_ns_sw"]["value"]
    elif "C1_XtWX_ns_v4" in prims:
        C1 = prims["C1_XtWX_ns_v4"]["value"] * 2200 / 100
    else:
        C1 = prims.get("C1_XtWX_ns", {}).get("value", 1000)
    C2 = prims["C2_solve_ns"]["value"]
    n_groups = SW2D_N_GROUPS
    T = n_groups * (C1 + C2) * 1e-9
    return T, {"C1_sw": C1, "C2": C2}


# ---- Imports ----

def _import_swf():
    try:
        from groupby_regression_sliding_window import make_sliding_window_fit
    except ImportError:
        from dfextensions.groupby_regression.groupby_regression_sliding_window import make_sliding_window_fit
    return make_sliding_window_fit


def _import_v4():
    try:
        from groupby_regression_optimized import make_parallel_fit_v4
    except ImportError:
        from dfextensions.groupby_regression.groupby_regression_optimized import make_parallel_fit_v4
    return make_parallel_fit_v4


# ---- Fixtures ----

@pytest.fixture(scope="session")
def sw2d_fixture():
    """Fixture B (SW2D): 32×32 grid, ~97 rows/bin, 2 linear columns. n_rows=100000."""
    rng = np.random.default_rng(13_22)
    rows = []
    row_count = 0
    for ix in range(SW2D_NX):
        for iy in range(SW2D_NY):
            bin_idx = ix * SW2D_NY + iy
            # Last group gets remainder to hit exactly 100000
            if bin_idx < SW2D_N_GROUPS - 1:
                n_in_bin = SW2D_RPG
            else:
                n_in_bin = SW2D_N_ROWS - row_count
            for _ in range(n_in_bin):
                x1 = rng.normal()
                x2 = rng.normal()
                rows.append({"bin_x": ix, "bin_y": iy, "x1": x1, "x2": x2,
                             "y": 2.0 + 0.5 * x1 - 0.3 * x2 + rng.normal(0, 0.1)})
            row_count += n_in_bin
    df = pd.DataFrame(rows)
    df["bin_x"] = df["bin_x"].astype(np.int64)
    df["bin_y"] = df["bin_y"].astype(np.int64)
    return df


@pytest.fixture(scope="session")
def s2_fixture():
    """Fixture A (S2): 100K rows, 1000 groups, flat grouping."""
    rng = np.random.default_rng(42)
    n = S2_N_ROWS
    df = pd.DataFrame({
        "group": rng.integers(0, S2_N_GROUPS, size=n),
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
    })
    df["y"] = 2.0 + 0.5 * df["x1"] - 0.3 * df["x2"] + rng.normal(0, 0.1, size=n)
    df["group"] = df["group"].astype(np.int64)
    return df


@pytest.fixture(scope="session")
def primitives():
    return _load_primitives()


@pytest.fixture(scope="session")
def baseline():
    return _load_baseline()


# ---- K-Roofline Tests ----

@pytest.mark.slow
@pytest.mark.roofline
class TestRoofline_SWFit:

    def test_fit_regression_roofline(self, sw2d_fixture, primitives, baseline):
        """make_sliding_window_fit: K = T_observed / Σ(n_ops × t_primitive).
        K=1 means at-roofline. Current pandas overhead expected in K (~3-5×).
        """
        if primitives is None:
            pytest.skip("No primitives file")
        swf = _import_swf()
        df = sw2d_fixture

        call = lambda: swf(
            df=df, gb_columns=["bin_x", "bin_y"], fit_columns=["y"],
            linear_columns=["x1", "x2"],
            window_spec={"bin_x": SW2D_WINDOW_R, "bin_y": SW2D_WINDOW_R},
            algorithm='recompute', backend='numba', min_stat=5, suffix='_sw')

        n_warmup = _warmup_until_stable(call)
        T_exp, info = _T_expected_sw_fit(primitives["primitives"])

        t0 = time.perf_counter()
        call()
        T_obs = time.perf_counter() - t0

        K = T_obs / T_exp if T_exp > 0 else float('inf')
        thresh = _get_threshold(baseline, "test_fit_regression_roofline")
        print(f"\n  K={K:.2f} (T_obs={T_obs*1e3:.1f}ms, T_exp={T_exp*1e3:.2f}ms, "
              f"warmup={n_warmup}, n_nbr={info['n_neighbors_eff']}, "
              f"C1_sw={info['C1_sw']:.0f}ns)")
        if thresh:
            assert K <= thresh, f"K={K:.2f} > threshold={thresh:.1f}"
        else:
            print(f"  [CALIBRATION] No threshold — reporting only")

    def test_assign_bin_ids_modeled_roofline(self, sw2d_fixture, primitives, baseline):
        """_assign_bin_ids_fast: K vs M1+M3w model (~8 reads + ~5 writes)."""
        if primitives is None:
            pytest.skip("No primitives file")
        import cProfile, pstats, io
        swf = _import_swf()
        df = sw2d_fixture

        call = lambda: swf(
            df=df, gb_columns=["bin_x", "bin_y"], fit_columns=["y"],
            linear_columns=["x1", "x2"],
            window_spec={"bin_x": SW2D_WINDOW_R, "bin_y": SW2D_WINDOW_R},
            algorithm='recompute', backend='numba', min_stat=5, suffix='_sw')

        _warmup_until_stable(call)
        T_exp, _ = _T_expected_assign_bin_ids(primitives["primitives"])

        pr = cProfile.Profile()
        pr.enable()
        call()
        pr.disable()
        stats = pstats.Stats(pr, stream=io.StringIO())
        T_obs = 0.0
        for key, (cc, nc, tt, ct, callers) in stats.stats.items():
            if '_assign_bin_ids_fast' in str(key[2]):
                T_obs = ct
                break

        K = T_obs / T_exp if T_exp > 0 else float('inf')
        thresh = _get_threshold(baseline, "test_assign_bin_ids_modeled_roofline")
        print(f"\n  K={K:.2f} (T_obs={T_obs*1e3:.1f}ms, T_exp={T_exp*1e3:.2f}ms)")
        if thresh:
            assert K <= thresh, f"K={K:.2f} > threshold={thresh:.1f}"
        else:
            print(f"  [CALIBRATION] No threshold — reporting only")

    # test_counting_sort_modeled_roofline REMOVED per Sonnet29 review:
    # cProfile cumtime on numba @njit kernel is unreliable (same issue as
    # test_gather_rows_modeled_roofline removed in v1.8 per Sonnet27 P1-4).
    # T_exp=0.19ms is below timer resolution; cProfile reports 5-7ms
    # including JIT/cache overhead. Counting-sort regressions are caught
    # by test_fit_regression_roofline (counting sort is a sub-component).


@pytest.mark.slow
@pytest.mark.roofline
class TestRoofline_V4:

    def test_v4_modeled_roofline(self, s2_fixture, primitives, baseline):
        """make_parallel_fit_v4: K = T_observed / Σ(n_ops × t_primitive).
        Per v1.10 §6.2 step 4: backend='pyarrow' pinned to match M2+M3w model.
        """
        if primitives is None:
            pytest.skip("No primitives file")
        v4 = _import_v4()
        df = s2_fixture

        call = lambda: v4(
            df=df, gb_columns=["group"], fit_columns=["y"],
            linear_columns=["x1", "x2"], min_stat=5, suffix='_v4',
            backend='pyarrow')

        n_warmup = _warmup_until_stable(call)
        T_exp, info = _T_expected_v4(primitives["primitives"])

        t0 = time.perf_counter()
        call()
        T_obs = time.perf_counter() - t0

        K = T_obs / T_exp if T_exp > 0 else float('inf')
        thresh = _get_threshold(baseline, "test_v4_modeled_roofline")
        print(f"\n  K={K:.2f} (T_obs={T_obs*1e3:.1f}ms, T_exp={T_exp*1e3:.2f}ms, "
              f"warmup={n_warmup})")
        if thresh:
            assert K <= thresh, f"K={K:.2f} > threshold={thresh:.1f}"
        else:
            print(f"  [CALIBRATION] No threshold — reporting only")

    def test_fit_kernel_modeled_roofline(self, sw2d_fixture, primitives, baseline):
        """SW fit pipeline with OLS-only denominator (C1+C2).

        NOTE: This test measures the FULL make_sliding_window_fit pipeline
        (T_observed includes gather, sort, bin-id, output assembly) but uses
        only C1+C2 as the denominator. K is therefore NOT interpretable as
        kernel efficiency alone — it measures "how many OLS-equivalents does
        the full pipeline cost." A regression in ANY sub-function raises K.

        This is a documented deviation from §4.5 which specifies kernel-only
        T_observed via no-fit subtraction. The no-fit helper is deferred to
        a follow-up; this test provides regression detection in the interim.
        """
        if primitives is None:
            pytest.skip("No primitives file")
        swf = _import_swf()
        df = sw2d_fixture

        call = lambda: swf(
            df=df, gb_columns=["bin_x", "bin_y"], fit_columns=["y"],
            linear_columns=["x1", "x2"],
            window_spec={"bin_x": SW2D_WINDOW_R, "bin_y": SW2D_WINDOW_R},
            algorithm='recompute', backend='numba', min_stat=5, suffix='_sw')

        _warmup_until_stable(call)
        T_exp, info = _T_expected_fit_kernel(primitives["primitives"])

        t0 = time.perf_counter()
        call()
        T_obs = time.perf_counter() - t0

        K = T_obs / T_exp if T_exp > 0 else float('inf')
        thresh = _get_threshold(baseline, "test_fit_kernel_modeled_roofline")
        print(f"\n  K={K:.2f} (T_obs={T_obs*1e3:.1f}ms, T_exp={T_exp*1e3:.2f}ms, "
              f"C1_sw={info['C1_sw']:.0f}ns, C2={info['C2']:.0f}ns)")
        if thresh:
            assert K <= thresh, f"K={K:.2f} > threshold={thresh:.1f}"
        else:
            print(f"  [CALIBRATION] No threshold — reporting only")

    def test_v4_median_dispatch_count(self, s2_fixture, primitives, baseline):
        """V4 np.median call count ≤ 1000 per §6.3 (was 1.26M pre-F3).

        Per v1.10 §6.2: backend='pyarrow' pinned. Must run single-process.
        """
        import cProfile, pstats, io
        v4 = _import_v4()
        df = s2_fixture

        call = lambda: v4(
            df=df, gb_columns=["group"], fit_columns=["y"],
            linear_columns=["x1", "x2"], min_stat=5, suffix='_v4',
            backend='pyarrow')

        _warmup_until_stable(call)

        pr = cProfile.Profile()
        pr.enable()
        call()
        pr.disable()
        stats = pstats.Stats(pr, stream=io.StringIO())
        n_median = 0
        for key, (cc, nc, tt, ct, callers) in stats.stats.items():
            if 'median' in str(key[2]):
                n_median += nc

        print(f"\n  np.median ncalls={n_median} (threshold ≤1000 per §6.3)")
        assert n_median <= 1000, (
            f"V4 made {n_median} np.median calls (expected ≤1000 per §6.3)."
        )


@pytest.mark.slow
@pytest.mark.roofline
class TestRoofline_Baseline:

    def test_baseline_file_loaded(self, baseline):
        if baseline is None:
            pytest.skip("No baseline — calibration mode")
        expected = {"test_fit_regression_roofline", "test_v4_modeled_roofline",
                    "test_assign_bin_ids_modeled_roofline",
                    "test_fit_kernel_modeled_roofline",
                    "test_v4_median_dispatch_count"}
        assert set(baseline["thresholds"].keys()) >= expected

    def test_baseline_self_consistent(self, baseline):
        if baseline is None:
            pytest.skip("No baseline — calibration mode")
        for test_name, t in baseline.get("thresholds", {}).items():
            if "K_floor" not in t:
                continue
            k_floor = t["K_floor"]
            for mname, mdata in baseline.get("machines", {}).items():
                k_entry = mdata.get("test_K_measured", {}).get(test_name)
                if k_entry is None:
                    continue
                k_mad = k_entry.get("K_mad", 0)
                sigma = 1.4826 * k_mad
                expected = k_floor + 6 * sigma
                stored = t.get(f"K_threshold_{mname}")
                if stored is not None:
                    assert abs(stored - expected) <= 1.0, (
                        f"{test_name}/{mname}: {stored} != {expected:.1f}")

    def test_baseline_machine_matches_runtime(self, baseline):
        """Runtime machine must exist in baseline (prevents cross-machine K mismatch)."""
        if baseline is None:
            pytest.skip("No baseline — calibration mode")
        machine = os.environ.get("GBAI_ROOFLINE_MACHINE",
                                 socket.gethostname().split('.')[0])
        assert machine in baseline.get("machines", {}), (
            f"Runtime machine '{machine}' not in baseline machines "
            f"{list(baseline.get('machines', {}).keys())}. "
            f"Set GBAI_ROOFLINE_MACHINE or regenerate baseline for this machine."
        )

    def test_baseline_threading_matches_runtime(self, baseline):
        """Calibration threading config must match runtime (v1.12 §6.1).

        Catches: calibration ran without NUMBA_THREADING_LAYER=omp → prange
        kernels ran serial → K_calibration 5× K_test → thresholds useless.
        Uses 'NOT_SET' sentinel on both sides per Sonnet26 v1.11 P1-2 fix.
        """
        if baseline is None:
            pytest.skip("No baseline — calibration mode")
        machine = os.environ.get("GBAI_ROOFLINE_MACHINE",
                                 socket.gethostname().split('.')[0])
        if machine not in baseline.get("machines", {}):
            return  # machine-match test handles this
        mdata = baseline["machines"][machine]
        baseline_layer = mdata.get("numba_threading_layer", "NOT_SET")
        baseline_threads = mdata.get("numba_num_threads")
        runtime_layer = os.environ.get("NUMBA_THREADING_LAYER", "NOT_SET")
        try:
            import numba
            runtime_threads = numba.get_num_threads()
        except (ImportError, AttributeError):
            runtime_threads = None
        assert baseline_layer == runtime_layer, (
            f"Baseline calibrated with NUMBA_THREADING_LAYER={baseline_layer!r}, "
            f"runtime is {runtime_layer!r}. Re-calibrate or set env var. "
            f"'NOT_SET' means env var was unset; both sides must agree."
        )
        if baseline_threads is not None and runtime_threads is not None:
            assert baseline_threads == runtime_threads, (
                f"Baseline calibrated with {baseline_threads} threads, "
                f"runtime has {runtime_threads}."
            )

    def test_baseline_phase_current(self, baseline):
        if baseline is None:
            pytest.skip("No baseline — calibration mode")
        from datetime import datetime
        cal_date = baseline.get("calibration_date", "2020-01-01")
        try:
            days = (datetime.now() - datetime.strptime(cal_date, "%Y-%m-%d")).days
            if days > 90:
                warnings.warn(f"Baseline is {days} days old")
        except ValueError:
            pass
