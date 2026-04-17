#!/usr/bin/env python3
"""Phase 13.18.GB Demo 1: PyROOT end-to-end parity check.

Workflow:
  1. Build a small synthetic dfGB in Python (intercept + 1 predictor, 2D grid)
  2. Dump to .root via dfGB_to_root.py (with metadata sidecar)
  3. Load in C++ via GBE::load_model_from_metadata
  4. Evaluate lookup + linear via the auto-generated GBE::eval_<name> stub
  5. Compare C++ predictions with Python reference at rtol=1e-12

Run on alma2:
  cd cpp
  make libGroupByRegressionEvaluator.so
  python3 demo/demo_pyroot.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dfGB_to_root import dfGB_to_root


def main():
    # --- 1. Build synthetic dfGB ---
    grid_x, grid_y = 4, 3
    rows = []
    for ix in range(grid_x):
        for iy in range(grid_y):
            rows.append({
                "sector": ix * 10,  # non-compact natural labels
                "padRow": iy * 5,
                "driftV_intercept_sw": 100.0 + ix * 10 + iy,
                "driftV_slope_spaceCharge_sw": 0.5 + ix * 0.1 - iy * 0.05,
            })
    df = pd.DataFrame(rows)
    for col in ["sector", "padRow"]:
        df[col] = df[col].astype(np.int64)

    tmp = Path(tempfile.mkdtemp()) / "demo_model.root"
    dfGB_to_root(df, tmp, tree_name="dfGB",
                 group_columns=["sector", "padRow"],
                 predictor_columns=["spaceCharge"],
                 targets=["driftV"], suffix="_sw", fit_intercept=True)
    print(f"Model written to: {tmp}")

    # --- 2. Load in C++ ---
    import ROOT
    lib = str(Path(__file__).resolve().parent.parent / "libGroupByRegressionEvaluator.so")
    rc = ROOT.gSystem.Load(lib)
    assert rc >= 0, f"Load failed: {rc}"

    # Option A: metadata sidecar
    ok = ROOT.GBE.load_model_from_metadata("driftV_demo", str(tmp),
                                            "dfGB", "lookup", "nan")
    assert ok, "load_model_from_metadata failed"
    print("Model loaded (lookup mode)")

    # --- 3. Scalar evaluation via auto-generated stub ---
    print("\n--- Scalar lookup parity ---")
    for _, row in df.iterrows():
        sx, pr = int(row.sector), int(row.padRow)
        sc = 1.5  # test predictor value
        cpp_pred = ROOT.GBE.eval_driftV_demo(float(sx), float(pr), sc)
        py_pred = row.driftV_intercept_sw + row.driftV_slope_spaceCharge_sw * sc
        match = abs(cpp_pred - py_pred) < 1e-12
        status = "✓" if match else "FAIL"
        print(f"  sector={sx:2d} padRow={pr:2d} sc={sc}: "
              f"C++={cpp_pred:.6f} Py={py_pred:.6f} {status}")
        assert match, f"Parity failure at ({sx},{pr})"

    # --- 4. Out-of-grid → NaN ---
    import math
    nan_pred = ROOT.GBE.eval_driftV_demo(999.0, 0.0, 1.0)
    assert math.isnan(nan_pred), f"out-of-grid should be NaN, got {nan_pred}"
    print(f"  out-of-grid (999,0,1.0) → NaN ✓")

    # --- 5. Linear mode ---
    ROOT.GBE.clear_models()
    ok = ROOT.GBE.load_model_from_metadata("driftV_lin", str(tmp),
                                            "dfGB", "linear", "nan")
    assert ok, "load_model_from_metadata (linear) failed"
    print("\n--- Scalar linear parity ---")
    # At integer grid points, linear == lookup
    for _, row in df.iterrows():
        sx, pr = int(row.sector), int(row.padRow)
        sc = 2.0
        cpp_pred = ROOT.GBE.eval_driftV_lin(float(sx), float(pr), sc)
        py_pred = row.driftV_intercept_sw + row.driftV_slope_spaceCharge_sw * sc
        match = abs(cpp_pred - py_pred) < 1e-12
        status = "✓" if match else "FAIL"
        print(f"  sector={sx:2d} padRow={pr:2d} sc={sc}: "
              f"C++={cpp_pred:.6f} Py={py_pred:.6f} {status}")
        assert match, f"Linear parity failure at ({sx},{pr})"

    ROOT.GBE.clear_models()
    print("\n=== Demo 1 (PyROOT parity): ALL PASSED ===")


if __name__ == "__main__":
    main()
