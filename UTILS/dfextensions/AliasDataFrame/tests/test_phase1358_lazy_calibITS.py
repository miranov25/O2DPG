"""
Phase 13.58.ADF — real-data lazy invariance on calibITS (and calibTRD-ready).

This is the real-data counterpart to the synthetic invariance suite. It reads an actual
distributable ADF calibration file (calibITS.root, ~4 MB, main tree 'AlignITS5' with two
subframes) lazily and eagerly and asserts, on real branches, that:
  (a) read_tree_lazy loads nothing at construction,
  (b) a real draw loads EXACTLY the branches it needs (and no others), and
  (c) the lazy stats equal the eager stats.

Unlike the time-series gallery double-run (validate_lazy_vs_eager / test_phase1358_gallery_lazy),
this does NOT use the time-series build_adf setup (no timeMS/sector) — calibITS has a
different schema. It needs dfdraw and the fixture file; it SKIPS cleanly otherwise. The
fixture is small enough to commit (tests/data/calibITS.root).

To run, place the file at one of the candidate paths below or set ADF_CALIB_ITS:
    ADF_CALIB_ITS=/path/to/calibITS.root pytest tests/test_phase1358_lazy_calibITS.py -v
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest
import uproot

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AliasDataFrame import AliasDataFrame  # noqa: E402

pytest.importorskip("dfextensions.dfdraw", reason="dfdraw not importable; real-data lazy gate skipped")

_HERE = os.path.dirname(os.path.abspath(__file__))
_CANDIDATES = [
    os.environ.get("ADF_CALIB_ITS"),
    os.path.join(_HERE, "data", "calibITS.root"),
    os.path.join(_HERE, "..", "tutorials", "data", "calibITS.root"),
]
CALIB_ITS = next((p for p in _CANDIDATES if p and os.path.exists(p)), None)

# Real calibITS schema (main tree 'AlignITS5', 28,956 entries).
TREE = "AlignITS5"
X = "firstTForbit"
Y = "dy_ITS5T_rms_AITS5"
Y2 = "dy_ITS5T_intercept_AITS5"
CAT = "staveITS"          # int8 category, good for group_by


def _eager(path, tree):
    """Eager reference: full uproot read of the main tree into an ADF (ROOT-free)."""
    arr = uproot.open(path)[tree].arrays(library="np")
    return AliasDataFrame(pd.DataFrame({k: np.asarray(v) for k, v in arr.items()}))


def _cmp_stats(sl, se, label):
    compared = 0
    for k in (set(sl) & set(se)):
        try:
            a = np.asarray(sl[k], dtype=float)
            b = np.asarray(se[k], dtype=float)
        except (TypeError, ValueError):
            continue
        if a.shape == b.shape and a.size:
            assert np.allclose(a, b, equal_nan=True), f"{label}: stat '{k}' lazy != eager"
            compared += 1
    assert compared > 0, f"{label}: no comparable numeric stats"


pytestmark = pytest.mark.skipif(
    CALIB_ITS is None,
    reason="calibITS.root not found; set ADF_CALIB_ITS or place it at tests/data/calibITS.root",
)


def test_calibITS_lazy_construct_loads_nothing():
    lazy = AliasDataFrame.read_tree_lazy(CALIB_ITS, TREE)
    assert lazy._lazy_reader.loaded_branches == set()
    assert {X, Y, CAT} <= lazy._lazy_reader.available_branches


@pytest.mark.invariance
def test_calibITS_profile_exact_load_and_equiv():
    """Real profile draw: loads exactly {Y, X}, lazy stats == eager stats."""
    lazy = AliasDataFrame.read_tree_lazy(CALIB_ITS, TREE)
    eager = _eager(CALIB_ITS, TREE)
    sl = lazy.draw(expr=f"{Y}:{X}", type="profile", bins=20, return_data=True)
    se = eager.draw(expr=f"{Y}:{X}", type="profile", bins=20, return_data=True)
    assert lazy._lazy_reader.loaded_branches == {Y, X}
    _cmp_stats(sl[2], se[2], "calibITS profile")
    assert np.array_equal(np.asarray(lazy.df[Y]), np.asarray(eager.df[Y]))


@pytest.mark.invariance
def test_calibITS_groupby_exact_load_and_equiv():
    """Real group_by draw on a category branch: loads exactly {Y, X, CAT}."""
    lazy = AliasDataFrame.read_tree_lazy(CALIB_ITS, TREE)
    eager = _eager(CALIB_ITS, TREE)
    sl = lazy.draw(expr=f"{Y}:{X}", type="profile", bins=20, group_by=CAT, return_data=True)
    se = eager.draw(expr=f"{Y}:{X}", type="profile", bins=20, group_by=CAT, return_data=True)
    assert lazy._lazy_reader.loaded_branches == {Y, X, CAT}
    _cmp_stats(sl[2], se[2], "calibITS group_by")


@pytest.mark.invariance
def test_calibITS_hist_exact_load():
    """Real hist draw: loads exactly {Y2}; decoys (other branches) do not load."""
    lazy = AliasDataFrame.read_tree_lazy(CALIB_ITS, TREE)
    lazy.draw(expr=Y2, type="hist", bins=30)
    assert lazy._lazy_reader.loaded_branches == {Y2}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
