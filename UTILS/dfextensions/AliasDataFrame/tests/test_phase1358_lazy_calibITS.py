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


@pytest.mark.invariance
def test_calibITS_subframe_column_lazy_draw():
    """T11 — subframe-column lazy draw on real data, robust to the fixture's metadata.

    calibITS has two ADF subframes ('R', 'AlignDzITS5'). read_tree_lazy recovers them from
    the file's metadata. The outcome depends on what the fixture carries:

      * Full metadata (subframe index columns present, e.g. the original calibITS): the
        subframes register as lazy subframes, and a subframe-column lazy draw works —
        materialized on demand via ensure_subframe, loading the main-tree X branch
        (Phase 13.58 subframe-draw orchestration).

      * Names-only recovery (e.g. a uproot-rewritten slim that dropped the UserInfo): no
        index columns, so the subframes are NOT registered, and a subframe-column draw must
        FAIL LOUD (clear error), never silently render an empty figure.

    The test asserts whichever contract applies, so it is correct on both the original file
    and the committed slim.
    """
    lazy = AliasDataFrame.read_tree_lazy(CALIB_ITS, TREE)
    meta = getattr(lazy._lazy_reader, "adf_metadata", None) or {}
    assert set(meta.get("subframes") or []) >= {"R", "AlignDzITS5"}   # names always recovered
    lazy.draw_lazy = True
    expr = "AlignDzITS5.dz_ITS5T_rms_AITS5:firstTForbit"

    if "AlignDzITS5" in set(getattr(lazy, "lazy_subframes", [])):
        # registered (index columns present) -> subframe-column lazy draw works
        lazy.draw(expr=expr, type="profile", bins=10)
        assert "firstTForbit" in lazy._lazy_reader.loaded_branches
        assert "AlignDzITS5" in set(lazy.list_subframes())            # materialized on demand
    else:
        # names-only recovery (no index columns) -> not registered -> fails loud
        with pytest.raises(ValueError):
            lazy.draw(expr=expr, type="profile", bins=10)


@pytest.mark.invariance
def test_calibITS_subframe_lazy_eager_value_parity():
    """PHASE_13_67 (architect-requested): loading a SUBFRAME column lazily must give exactly
    the same values as the eager read. Only meaningful when the fixture carries full subframe
    metadata (index columns) so the subframe registers; the names-only slim skips."""
    lazy = AliasDataFrame.read_tree_lazy(CALIB_ITS, TREE)
    if "AlignDzITS5" not in set(getattr(lazy, "lazy_subframes", [])):
        pytest.skip("names-only fixture: subframes not registered; no subframe data to load")
    eager = AliasDataFrame.read_tree(CALIB_ITS, TREE)
    expr = "AlignDzITS5.dz_ITS5T_rms_AITS5"
    lazy_vals = np.asarray(lazy.eval(expr), dtype=float)
    eager_vals = np.asarray(eager.eval(expr), dtype=float)
    assert lazy_vals.shape == eager_vals.shape
    assert np.allclose(lazy_vals, eager_vals, equal_nan=True), \
        "subframe column: lazy load != eager load"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


# ── PHASE_13_67_ADF: back-compat — lazy read applies the same UserInfo metadata as eager ──
# Old files carry UserInfo written the old way. Rev 3.1 makes the lazy path APPLY it by
# default (0a). This guards, on real calibITS data, that (a) old data stays readable,
# (b) lazy aliases == eager read_tree aliases, (c) same branch universe, and (d) metadata
# application loads nothing at construction (INV-1). Automatic so it is not re-checked by hand.

@pytest.mark.invariance
def test_calibITS_lazy_applies_same_metadata_as_eager():
    lazy = AliasDataFrame.read_tree_lazy(CALIB_ITS, TREE)
    eager = AliasDataFrame.read_tree(CALIB_ITS, TREE)          # read_tree applies UserInfo
    # (a)+(b) aliases recovered and applied on the lazy path == eager
    assert set(lazy.aliases) == set(eager.aliases), (
        f"lazy aliases {set(lazy.aliases)} != eager {set(eager.aliases)}")
    # (c) same branch universe (lazy knows all branches without loading them)
    assert sorted(lazy.available_branches) == sorted(eager.df.columns)
    # (d) metadata application is lazy — nothing loaded at construction (INV-1)
    assert lazy._lazy_reader.loaded_branches == set()
    # subframe NAMES always recovered from UserInfo (registration as lazy_subframes
    # requires index columns — absent in the names-only slim fixture, present in the full
    # file; assert on the always-present recovered names, robust to both fixtures).
    _meta = getattr(lazy._lazy_reader, "adf_metadata", None) or {}
    assert {"R", "AlignDzITS5"} <= set(_meta.get("subframes") or [])


@pytest.mark.invariance
def test_calibITS_lazy_value_parity_with_metadata_eager():
    """INV-4(b): a lazily-loaded column's values equal the eager read_tree (metadata-bearing)
    read of the same column — metadata application does not corrupt data."""
    lazy = AliasDataFrame.read_tree_lazy(CALIB_ITS, TREE)
    eager = AliasDataFrame.read_tree(CALIB_ITS, TREE)
    lazy.ensure_columns([Y])
    assert np.array_equal(np.asarray(lazy.df[Y]), np.asarray(eager.df[Y]))
