#!/usr/bin/env python3
"""Phase 13.18.GB Turn 6 — alma2 PyROOT smoke test for Layer B.

Run on alma2 after `make libGroupByRegressionEvaluator.so` succeeds.
This is NOT a pytest file — it is a standalone script meant for
manual verification per the Turn 6 acceptance items in proposal v1.1
§7. It exercises:

  1. gSystem.Load() returns 0
  2. GBE namespace symbols are visible (has_model, list_models)
  3. dfGB_to_root.py round-trip: dump a Python evaluator's dfGB to
     .root with a sidecar, then load it back via load_model_from_metadata
  4. Both Option A (sidecar) and Option B (explicit schema) succeed
  5. Loaded model's grid_shape and populated_cells match the Python
     side (proves valid_mask + sparse->dense load worked)
  6. Error-suffix columns (*_err_sw etc.) survive load (P1-delta)

Usage on alma2:
    cd cpp
    make libGroupByRegressionEvaluator.so
    python3 tests/smoke_test_layer_b.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

CPP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CPP_ROOT))

from dfGB_to_root import dfGB_to_root  # noqa: E402

LIB_PATH = CPP_ROOT / "libGroupByRegressionEvaluator.so"


def banner(text: str) -> None:
    print(f"\n=== {text} ===")


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def main() -> None:
    if not LIB_PATH.exists():
        fail(f"library not found at {LIB_PATH}; "
             f"run `make libGroupByRegressionEvaluator.so` first")

    banner("Test 1: gSystem.Load returns 0")
    import ROOT
    rc = ROOT.gSystem.Load(str(LIB_PATH))
    print(f"  Load returned: {rc}")
    if rc < 0:
        fail("gSystem.Load failed (rc < 0)")

    banner("Test 2: GBE namespace visible")
    print(f"  GBE.has_model('nope'): {ROOT.GBE.has_model('nope')}")
    print(f"  GBE.list_models() initial: {list(ROOT.GBE.list_models())}")
    if ROOT.GBE.has_model("nope"):
        fail("has_model on unknown name should be False")

    banner("Test 3: dump fixture .root via dfGB_to_root")
    # Build a small 2D dfGB with intercept + 1 predictor + error columns
    df = pd.DataFrame({
        "bin_x": np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int64),
        "bin_y": np.array([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.int64),
        "y_intercept_fit":      np.linspace(-1.0, 1.0, 9),
        "y_slope_x_fit":        np.linspace(-0.5, 0.5, 9),
        # Unknown-suffix tolerance check (P1-delta)
        "y_intercept_err_fit":  np.linspace(0.01, 0.05, 9),
        "y_rmse_fit":           np.linspace(0.1,  0.5,  9),
        "y_n_fitted_fit":       np.array([10]*9, dtype=np.int64),
    })
    tmp = Path(tempfile.mkdtemp()) / "smoke.root"
    dfGB_to_root(
        df, tmp, tree_name="dfGB",
        group_columns=["bin_x", "bin_y"],
        predictor_columns=["x"], targets=["y"],
        suffix="_fit", fit_intercept=True,
    )
    print(f"  dumped fixture to: {tmp}")

    banner("Test 4: load_model_explicit (Option B)")
    ok = ROOT.GBE.load_model_explicit(
        "smoke_explicit",       # model name
        str(tmp),               # path
        "dfGB",                 # tree_name
        ROOT.std.vector("string")(["bin_x", "bin_y"]),
        ROOT.std.vector("string")(["x"]),
        ROOT.std.vector("string")(["y"]),
        "_fit",                 # suffix
        True,                   # fit_intercept
        "lookup",               # method
        "nan",                  # bounds
    )
    if not ok:
        fail("load_model_explicit returned false")
    print(f"  loaded; has_model('smoke_explicit'): "
          f"{ROOT.GBE.has_model('smoke_explicit')}")

    banner("Test 5: model grid + populated cells via get_model")
    ev = ROOT.GBE.get_model("smoke_explicit")
    if not ev:
        fail("get_model returned null")
    grid_shape = list(ev.grid_shape())
    populated  = ev.populated_cells()
    total      = ev.total_cells()
    print(f"  grid_shape    : {grid_shape}")
    print(f"  populated/total: {populated}/{total}")
    if grid_shape != [3, 3]:
        fail(f"grid_shape expected [3,3]; got {grid_shape}")
    if populated != 9 or total != 9:
        fail(f"populated/total expected 9/9; got {populated}/{total}")

    banner("Test 6: load_model_from_metadata (Option A)")
    ok = ROOT.GBE.load_model_from_metadata(
        "smoke_metadata",
        str(tmp),
        "dfGB",
        "lookup",
        "nan",
    )
    if not ok:
        fail("load_model_from_metadata returned false")
    ev2 = ROOT.GBE.get_model("smoke_metadata")
    if not ev2:
        fail("get_model('smoke_metadata') returned null")
    if list(ev2.grid_shape()) != [3, 3]:
        fail("Option A grid_shape mismatch")
    print("  Option A load OK; both models visible:")
    print(f"  list_models(): {list(ROOT.GBE.list_models())}")

    banner("Test 7: unload / clear")
    if not ROOT.GBE.unload_model("smoke_explicit"):
        fail("unload_model returned false")
    if ROOT.GBE.has_model("smoke_explicit"):
        fail("model still present after unload")
    ROOT.GBE.clear_models()
    if len(list(ROOT.GBE.list_models())) != 0:
        fail("registry not empty after clear_models")

    banner("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
