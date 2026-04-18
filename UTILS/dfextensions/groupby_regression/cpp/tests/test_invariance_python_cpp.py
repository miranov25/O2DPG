#!/usr/bin/env python3
"""Phase 13.18.GB — Python ↔ C++ Invariance Test.

Runs in a single Python session on alma2. Demonstrates equivalence
between the production Python `GroupByRegressionEvaluator` and the
C++ `GBE::eval_<model>` (via PyROOT) across multiple evaluation
settings.

Settings matrix tested:
  method:        lookup, linear
  bounds:        nan, clamp
  fit_intercept: True (always)
  dimensions:    1D, 2D

For each setting combination:
  1. Build synthetic dfGB DataFrame
  2. Construct the Python GroupByRegressionEvaluator via from_dfGB
  3. Dump dfGB to .root via dfGB_to_root.py
  4. Load in C++ via GBE::load_model_from_metadata (PyROOT)
  5. Evaluate BOTH at the same query positions
  6. Compare with tolerance: rtol=1e-12 for lookup, rtol=1e-10 for linear

Usage on alma2:
  cd groupby_regression
  python3 cpp/tests/test_invariance_python_cpp.py

Requires: ROOT, PyROOT, groupby_regression Python package, the .so built.
"""
from __future__ import annotations

import math
import sys
import tempfile
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

# --- Setup paths ---
REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # groupby_regression/
CPP_ROOT = Path(__file__).resolve().parent.parent          # cpp/
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(CPP_ROOT))

from dfGB_to_root import dfGB_to_root
from groupby_regression_evaluator import GroupByRegressionEvaluator


def build_synthetic_dfGB(
    *,
    n_bins_x: int = 4,
    n_bins_y: int | None = None,
    predictor_columns: list[str],
    targets: list[str],
    suffix: str = "_fit",
    fit_intercept: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a synthetic dfGB DataFrame with known coefficient values."""
    rng = np.random.default_rng(seed)
    rows = []
    dims_x = list(range(n_bins_x))
    dims_y = list(range(n_bins_y)) if n_bins_y else [None]

    for ix in dims_x:
        for iy in dims_y:
            row = {"bin_x": ix}
            if iy is not None:
                row["bin_y"] = iy
            for t in targets:
                if fit_intercept:
                    row[f"{t}_intercept{suffix}"] = float(rng.uniform(-5, 5))
                for p in predictor_columns:
                    row[f"{t}_slope_{p}{suffix}"] = float(rng.uniform(-1, 1))
            rows.append(row)

    df = pd.DataFrame(rows)
    for col in ["bin_x"] + (["bin_y"] if n_bins_y else []):
        df[col] = df[col].astype(np.int64)
    return df


def evaluate_python(
    dfGB: pd.DataFrame,
    group_columns: list[str],
    predictor_columns: list[str],
    targets: list[str],
    suffix: str,
    fit_intercept: bool,
    method: str,
    bounds: str,
    query_positions: list[list[float]],
    predictor_values: list[list[float]],
) -> dict[str, list[float]]:
    """Evaluate using the production Python GroupByRegressionEvaluator."""
    ev = GroupByRegressionEvaluator.from_dfGB(
        dfGB,
        group_columns=group_columns,
        predictor_columns=predictor_columns,
        targets=targets,
        suffix=suffix,
    )
    results = {}
    for t in targets:
        preds = []
        for qi, pos in enumerate(query_positions):
            pv = predictor_values[qi] if predictor_values else []
            pred = ev.evaluate(
                positions={gc: [pos[d]] for d, gc in enumerate(group_columns)},
                predictors={p: [pv[pi]] for pi, p in enumerate(predictor_columns)} if predictor_columns else {},
                method=method,
                bounds=bounds,
            )
            val = pred[t]
            preds.append(float(np.atleast_1d(val)[0]))
        results[t] = preds
    return results


def evaluate_cpp(
    model_name: str,
    query_positions: list[list[float]],
    predictor_values: list[list[float]],
    n_gc: int,
) -> list[float]:
    """Evaluate using C++ GBE::eval_<model_name> via PyROOT."""
    import ROOT
    eval_fn = getattr(ROOT.GBE, f"eval_{model_name}")
    results = []
    for qi, pos in enumerate(query_positions):
        pv = predictor_values[qi] if predictor_values else []
        args = [float(x) for x in pos] + [float(x) for x in pv]
        pred = eval_fn(*args)
        results.append(pred)
    return results


def run_invariance_test(
    *,
    label: str,
    group_columns: list[str],
    predictor_columns: list[str],
    targets: list[str],
    suffix: str,
    fit_intercept: bool,
    method: str,
    bounds: str,
    dfGB: pd.DataFrame,
    query_positions: list[list[float]],
    predictor_values: list[list[float]],
    rtol: float,
    model_counter: list[int],
) -> tuple[int, int]:
    """Run one invariance comparison. Returns (n_tested, n_failed)."""
    import ROOT

    model_name = f"inv_{model_counter[0]}"
    model_counter[0] += 1

    # Dump to .root
    tmp = Path(tempfile.mkdtemp()) / f"{model_name}.root"
    dfGB_to_root(
        dfGB, tmp, tree_name="dfGB",
        group_columns=group_columns,
        predictor_columns=predictor_columns,
        targets=targets,
        suffix=suffix,
        fit_intercept=True,
    )

    # Load in C++
    ok = ROOT.GBE.load_model_from_metadata(
        model_name, str(tmp), "dfGB", method, bounds)
    if not ok:
        print(f"  FAIL: load_model_from_metadata failed")
        return len(query_positions), len(query_positions)

    # Evaluate Python
    py_results = evaluate_python(
        dfGB, group_columns, predictor_columns, targets, suffix,
        True, method, bounds, query_positions, predictor_values)

    # Evaluate C++
    cpp_results = evaluate_cpp(
        model_name, query_positions, predictor_values,
        len(group_columns))

    # Compare
    n_tested = 0
    n_failed = 0
    for qi in range(len(query_positions)):
        py_val = py_results[targets[0]][qi]
        cpp_val = cpp_results[qi]
        n_tested += 1

        py_nan = math.isnan(py_val)
        cpp_nan = math.isnan(cpp_val)

        if py_nan and cpp_nan:
            status = "NaN==NaN ✓"
        elif py_nan or cpp_nan:
            status = f"NaN MISMATCH py={py_val} cpp={cpp_val}"
            n_failed += 1
        else:
            denom = max(abs(py_val), abs(cpp_val), 1e-300)
            rel = abs(py_val - cpp_val) / denom
            if rel <= rtol:
                status = f"✓ (rtol={rel:.1e})" if rel > 0 else "✓ (exact)"
            else:
                status = f"FAIL rtol={rel:.1e} > {rtol:.0e}"
                n_failed += 1

        print(f"    q{qi} pos={query_positions[qi]} "
              f"py={py_val:.10g} cpp={cpp_val:.10g} {status}")

    ROOT.GBE.unload_model(model_name)
    return n_tested, n_failed


def main():
    # Load ROOT + library
    import ROOT
    lib = str(CPP_ROOT / "libGroupByRegressionEvaluator.so")
    rc = ROOT.gSystem.Load(lib)
    if rc < 0:
        print(f"FAIL: cannot load {lib} (rc={rc})")
        sys.exit(1)

    model_counter = [0]
    total_tested = 0
    total_failed = 0

    settings = list(product(
        ["lookup", "linear"],   # method
        ["nan", "clamp"],       # bounds
    ))

    print("=" * 70)
    print("Phase 13.18.GB — Python ↔ C++ Invariance Test")
    print("=" * 70)

    # ---- 1D tests ----
    for method, bounds in settings:
        label = f"1D method={method} bounds={bounds}"
        print(f"\n--- {label} ---")

        pred_cols = ["x"]
        dfGB = build_synthetic_dfGB(
            n_bins_x=4,
            predictor_columns=["x"],
            targets=["y"],
            fit_intercept=True,
            seed=hash(label) & 0xFFFFFFFF,
        )

        if method == "lookup":
            qp = [[0.0], [1.0], [2.0], [3.0]]
            rtol = 1e-12
        else:
            qp = [[0.0], [0.5], [1.5], [2.75], [3.0]]
            rtol = 1e-10

        if bounds == "nan":
            qp.append([-1.0])  # out-of-grid -> NaN
            qp.append([5.0])   # out-of-grid -> NaN
        else:  # clamp
            qp.append([-1.0])  # clamps to 0
            qp.append([5.0])   # clamps to 3

        pv = [[1.5]] * len(qp)

        nt, nf = run_invariance_test(
            label=label,
            group_columns=["bin_x"],
            predictor_columns=["x"],
            targets=["y"],
            suffix="_fit",
            fit_intercept=True,
            method=method,
            bounds=bounds,
            dfGB=dfGB,
            query_positions=qp,
            predictor_values=pv,
            rtol=rtol,
            model_counter=model_counter,
        )
        total_tested += nt
        total_failed += nf

    # ---- 2D tests ----
    for method, bounds in settings:
        label = f"2D method={method} bounds={bounds}"
        print(f"\n--- {label} ---")

        dfGB = build_synthetic_dfGB(
            n_bins_x=3,
            n_bins_y=3,
            predictor_columns=["x"],
            targets=["y"],
            fit_intercept=True,
            seed=hash(label) & 0xFFFFFFFF,
        )

        if method == "lookup":
            qp = [[0.0, 0.0], [1.0, 2.0], [2.0, 2.0]]
            rtol = 1e-12
        else:
            qp = [[0.0, 0.0], [0.5, 1.5], [1.5, 1.5], [2.0, 2.0]]
            rtol = 1e-10

        if bounds == "nan":
            qp.append([-1.0, 0.0])  # out-of-grid
        else:
            qp.append([-1.0, -1.0])  # clamps to (0,0)

        pv = [[2.0]] * len(qp)

        nt, nf = run_invariance_test(
            label=label,
            group_columns=["bin_x", "bin_y"],
            predictor_columns=["x"],
            targets=["y"],
            suffix="_fit",
            fit_intercept=True,
            method=method,
            bounds=bounds,
            dfGB=dfGB,
            query_positions=qp,
            predictor_values=pv,
            rtol=rtol,
            model_counter=model_counter,
        )
        total_tested += nt
        total_failed += nf

    # ---- Summary ----
    print("\n" + "=" * 70)
    print(f"INVARIANCE TEST SUMMARY")
    print(f"  Settings tested:  {len(settings) * 2} "
          f"(4 × 1D + 4 × 2D)")
    print(f"  Query points:     {total_tested}")
    print(f"  Failed:           {total_failed}")
    print(f"  Tolerances:       lookup rtol=1e-12, linear rtol=1e-10")
    if total_failed == 0:
        print(f"\n  ✅ ALL INVARIANCE TESTS PASSED")
    else:
        print(f"\n  ❌ {total_failed} FAILURES")
    print("=" * 70)

    ROOT.GBE.clear_models()
    sys.exit(1 if total_failed > 0 else 0)


if __name__ == "__main__":
    main()
