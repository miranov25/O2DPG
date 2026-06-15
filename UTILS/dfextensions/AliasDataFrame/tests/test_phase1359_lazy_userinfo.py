"""
test_phase1359_lazy_userinfo.py — Phase 13.59.ADF / BUG_20260613_lazy_UserInfo_gap.

AD-3/13.59.ADF metadata read precedence and the lazy-path UserInfo fix.

Self-contained: NO external fixtures (no calibITS.root / calibTRD.root, no env vars).
Test data is generated in-test, following the I-series pattern (build an ADF, write it,
read it back eager vs lazy). Import relies on tests/conftest.py adding the package dir to
sys.path; the parent insert below mirrors test_I10_lazy_eager_invariance.py.

uproot cannot write TTree UserInfo, so a UserInfo fixture must be produced by ROOT via
export_tree -> those tests are ROOT-dependent (skip without ROOT), exactly like the
existing I-series UserInfo tests. The standalone-key path is uproot-writable, so the
key-path invariance runs everywhere (the registration contract that broke).

Groups
------
INVARIANCE (@pytest.mark.invariance):
  - key-path registration (ROOT-FREE): read_tree_lazy on a uproot-written file whose
    subframes are described by the <tree>__adfmeta__ key exposes exactly those subframes.
    This is the contract that broke (lazy_subframes was []). FAILS on pre-fix code.
  - lazy ≡ eager subframe set (ROOT): read_tree (eager) and read_tree_lazy of the same
    export_tree-written UserInfo file expose the same subframes; the lazy subframe is
    accessible (ensure_subframe loads rows), not just named.
  - ROOT ≡ uproot UserInfo (AD-3 levels 1 vs 2, ROOT): the resolver recovers identical
    subframes/indices via ROOT (prefer_root=True) and uproot (prefer_root=False).

UNIT (ROOT-free): resolver precedence levels 3 (key) and 4 (names-only + indicator).
"""
import os
import sys
import warnings
import tempfile

import numpy as np
import pandas as pd
import pytest
import uproot

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AliasDataFrame import AliasDataFrame  # noqa: E402
from adf_metadata_compat import read_adf_metadata, write_adf_metadata_key  # noqa: E402

try:
    import ROOT
    _HAS_ROOT = ROOT is not None
except Exception:
    _HAS_ROOT = False

_requires_root = pytest.mark.skipif(not _HAS_ROOT, reason="ROOT not available (UserInfo write needs ROOT)")


# --------------------------- fixtures (self-generated) ---------------------------

@pytest.fixture
def keyfile(tmp_path):
    """ROOT-FREE: uproot file with main + sibling subframe trees + <tree>__adfmeta__ key."""
    path = str(tmp_path / "p1359_key.root")
    with uproot.recreate(path) as fo:
        fo["tree"] = {"run": np.arange(10, dtype="i4"), "x": np.random.rand(10)}
        fo["tree__subframe__A"] = {"run": np.arange(5, dtype="i4"), "a": np.random.rand(5)}
        fo["tree__subframe__B"] = {"run": np.arange(5, dtype="i4"),
                                   "sector": np.arange(5, dtype="i4"), "b": np.random.rand(5)}
        write_adf_metadata_key(fo, "tree", {"subframes": ["A", "B"],
                                            "subframe_indices": {"A": ["run"],
                                                                 "B": ["run", "sector"]}})
    return path


@pytest.fixture
def userinfo_file(tmp_path):
    """ROOT: build an ADF with two subframes and export_tree (writes UserInfo via ROOT)."""
    main = AliasDataFrame(pd.DataFrame({"run": np.arange(10), "x": np.random.rand(10)}))
    sfA = AliasDataFrame(pd.DataFrame({"run": np.arange(10), "a": np.random.rand(10)}))
    sfB = AliasDataFrame(pd.DataFrame({"run": np.arange(10), "b": np.random.rand(10)}))
    main.register_subframe("A", sfA, index_columns=["run"])
    main.register_subframe("B", sfB, index_columns=["run"])
    path = str(tmp_path / "p1359_userinfo.root")
    main.export_tree(path, treename="tree")
    return path


# --------------------------- INVARIANCE ---------------------------

@pytest.mark.invariance
def test_keypath_lazy_registration_invariance(keyfile):
    """ROOT-FREE: lazy path exposes the subframes described by the standalone key.

    Contract that broke (BUG_20260613): lazy_subframes was [] regardless of metadata.
    FAILS on pre-fix code. Runs everywhere (no ROOT, no external fixtures).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adf = AliasDataFrame.read_tree_lazy(keyfile, "tree")
    assert set(adf.lazy_subframes) == {"A", "B"}
    # ACCESS, not just the name (Sonnet2/Claude37/GPT15): load and check rows
    adf.ensure_subframe("A")
    assert "A" in adf.loaded_subframes
    sub = adf.get_subframe("A")
    assert len(sub) > 0 and len(sub.columns) > 0


@pytest.mark.invariance
@_requires_root
def test_lazy_eager_subframe_invariance(userinfo_file):
    """ROOT: eager and lazy reads of the same UserInfo file expose the same subframe set,
    and the lazy subframe is accessible (loads real rows), not merely named (Sonnet6)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eager = AliasDataFrame.read_tree(userinfo_file, "tree")
        lazy = AliasDataFrame.read_tree_lazy(userinfo_file, "tree")
    assert set(lazy.lazy_subframes) == set(eager.loaded_subframes) == {"A", "B"}
    # ACCESS, not just the name
    lazy.ensure_subframe("A")
    assert "A" in lazy.loaded_subframes
    sub = lazy.get_subframe("A")
    assert len(sub) > 0 and len(sub.columns) > 0


@pytest.mark.invariance
@_requires_root
def test_root_uproot_userinfo_invariance(userinfo_file):
    """ROOT: AD-3 levels 1 vs 2 — ROOT UserInfo read ≡ uproot UserInfo read."""
    via_root = read_adf_metadata(userinfo_file, "tree", prefer_root=True)
    via_uproot = read_adf_metadata(userinfo_file, "tree", prefer_root=False)
    assert via_root["_source"] == "root_userinfo"
    assert via_uproot["_source"] == "uproot_userinfo"
    assert sorted(via_root["subframes"]) == sorted(via_uproot["subframes"]) == ["A", "B"]
    assert via_root["subframe_indices"] == via_uproot["subframe_indices"]


# --------------------------- UNIT (ROOT-free) ---------------------------

def test_precedence_key_when_only_key():
    """AC-3 level 3: standalone key recovered when no UserInfo present."""
    tmp = tempfile.mktemp(suffix=".root")
    try:
        with uproot.recreate(tmp) as fo:
            fo["t"] = {"x": np.arange(5)}
            write_adf_metadata_key(fo, "t", {"subframes": ["K"],
                                             "subframe_indices": {"K": ["x"]}})
        m = read_adf_metadata(tmp, "t")
        assert m["_source"] == "key"
        assert m["subframes"] == ["K"]
    finally:
        os.path.exists(tmp) and os.remove(tmp)


def test_precedence_names_only_carries_indicator():
    """AC-3 level 4: names reconstruction is structure-only, flagged schema_source='names_only'."""
    tmp = tempfile.mktemp(suffix=".root")
    try:
        with uproot.recreate(tmp) as fo:
            fo["main"] = {"x": np.arange(3)}
            fo["main__subframe__SF"] = {"k": np.arange(3)}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            m = read_adf_metadata(tmp, "main")
            assert any("names_only" in str(x.message) for x in w)
        assert m["_source"] == "names"
        assert m["subframes"] == ["SF"]
        assert m.get("schema_source") == "names_only"
    finally:
        os.path.exists(tmp) and os.remove(tmp)


def test_names_only_no_registration():
    """Level 4 on the lazy path: a sibling-only file (no UserInfo, no key) warns and does
    NOT register unusable subframes (no indices). ROOT-free."""
    tmp = tempfile.mktemp(suffix=".root")
    try:
        with uproot.recreate(tmp) as fo:
            fo["main"] = {"x": np.arange(3, dtype="i4")}
            fo["main__subframe__SF"] = {"k": np.arange(3, dtype="i4")}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            adf = AliasDataFrame.read_tree_lazy(tmp, "main")
            assert any("not registered" in str(x.message) for x in w)
        assert list(adf.lazy_subframes) == []   # structure-only: not registered
    finally:
        os.path.exists(tmp) and os.remove(tmp)


@pytest.mark.invariance
@_requires_root
def test_conflict_userinfo_beats_key(userinfo_file):
    """AD-3 precedence: when both UserInfo and a standalone key are present, UserInfo wins."""
    with uproot.update(userinfo_file) as fo:
        write_adf_metadata_key(fo, "tree", {"subframes": ["DECOY"],
                                            "subframe_indices": {"DECOY": ["run"]}})
    m = read_adf_metadata(userinfo_file, "tree")
    assert m["_source"] in ("root_userinfo", "uproot_userinfo")   # UserInfo, not 'key'
    assert set(m["subframes"]) == {"A", "B"}                       # not DECOY


@pytest.mark.invariance
@_requires_root
def test_multientry_userinfo_loop(tmp_path):
    """Level 2 loops all UserInfo entries: a decoy first entry must not hide the ADF JSON."""
    import json as _json
    import array as _array
    path = str(tmp_path / "p1359_multi.root")
    f = ROOT.TFile.Open(path, "RECREATE")
    t = ROOT.TTree("tree", "tree")
    x = _array.array("f", [0.0])
    t.Branch("x", x, "x/F")
    for i in range(5):
        x[0] = float(i)
        t.Fill()
    ui = t.GetUserInfo()
    ui.Add(ROOT.TObjString("not-adf-json-decoy"))                 # decoy first
    ui.Add(ROOT.TObjString(_json.dumps({"subframes": ["M"],
                                        "subframe_indices": {"M": ["x"]}})))
    t.Write()
    f.Close()
    m = read_adf_metadata(path, "tree", prefer_root=False)        # force uproot level 2
    assert m["_source"] == "uproot_userinfo"
    assert m["subframes"] == ["M"]


def test_G2_dtype_recovery(tmp_path):
    """G2: AC-1 dtype recovery — resolver surfaces column_dtypes when present."""
    path = str(tmp_path / "g2.root")
    with uproot.recreate(path) as fo:
        fo["t"] = {"x": np.arange(5, dtype="i4")}
        write_adf_metadata_key(fo, "t", {
            "subframes": [], "subframe_indices": {},
            "column_dtypes": {"x": "int32", "y": "float64"},
        })
    m = read_adf_metadata(path, "t")
    assert m["column_dtypes"] == {"x": "int32", "y": "float64"}


def test_G3_alias_recovery(tmp_path):
    """G3: resolver surfaces aliases when present."""
    path = str(tmp_path / "g3.root")
    with uproot.recreate(path) as fo:
        fo["t"] = {"x": np.arange(5, dtype="i4")}
        write_adf_metadata_key(fo, "t", {
            "subframes": [], "subframe_indices": {},
            "aliases": {"x2": "x*2", "xsq": "x**2"},
        })
    m = read_adf_metadata(path, "t")
    assert m["aliases"] == {"x2": "x*2", "xsq": "x**2"}


def test_G5_registration_failure_path(tmp_path):
    """G5: metadata names a subframe whose sibling tree is missing -> warn and skip."""
    path = str(tmp_path / "g5.root")
    with uproot.recreate(path) as fo:
        fo["tree"] = {"run": np.arange(5, dtype="i4")}
        # NOTE: no 'tree__subframe__GHOST' sibling tree is written
        write_adf_metadata_key(fo, "tree", {"subframes": ["GHOST"],
                                            "subframe_indices": {"GHOST": ["run"]}})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        adf = AliasDataFrame.read_tree_lazy(path, "tree")
        assert any("could not register subframe 'GHOST'" in str(x.message) for x in w)
    assert "GHOST" not in adf.lazy_subframes


@_requires_root
def test_G4_malformed_userinfo_fallthrough(tmp_path):
    """G4: a malformed UserInfo entry warns and falls through to the standalone key."""
    import json as _json
    import array as _array
    path = str(tmp_path / "g4.root")
    f = ROOT.TFile.Open(path, "RECREATE")
    t = ROOT.TTree("tree", "tree")
    x = _array.array("f", [0.0])
    t.Branch("x", x, "x/F")
    for i in range(5):
        x[0] = float(i)
        t.Fill()
    t.GetUserInfo().Add(ROOT.TObjString("{this is not valid json"))   # malformed, no ADF entry
    t.Write()
    f.Close()
    # add a valid standalone key as the lower-precedence source
    with uproot.update(path) as fo:
        write_adf_metadata_key(fo, "tree", {"subframes": ["K"],
                                            "subframe_indices": {"K": ["x"]}})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = read_adf_metadata(path, "tree", prefer_root=False)        # force uproot path
        assert any("failed JSON parsing" in str(x.message) for x in w)
    assert m["_source"] == "key"                                      # fell through to the key
    assert m["subframes"] == ["K"]


# --- G1: real-fixture test, committed but skipped when fixtures are absent -----------
def _find_fixture(name, env):
    import os as _os
    for cand in (_os.environ.get(env), _os.path.join(_HERE_G1, name),
                 f"/mnt/user-data/uploads/{name}", _os.path.join(_os.getcwd(), name)):
        if cand and _os.path.exists(cand):
            return cand
    return None


_HERE_G1 = os.path.dirname(os.path.abspath(__file__))
_ITS = _find_fixture("calibITS.root", "ADF_ITS")


@pytest.mark.skipif(_ITS is None, reason="calibITS.root fixture not present (optional)")
def test_G1_real_fixture_recovery_ITS():
    """G1: real old-format calibITS recovery as a committed test (skips if file absent).

    ROOT-free (uproot level 2). Asserts subframes recovered from the real UserInfo.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adf = AliasDataFrame.read_tree_lazy(_ITS, "AlignITS5")
    assert sorted(adf.lazy_subframes) == ["AlignDzITS5", "R"]
    m = read_adf_metadata(_ITS, "AlignITS5", prefer_root=False)
    assert m["_source"] == "uproot_userinfo"
    assert m["subframe_indices"]["AlignDzITS5"] == ["staveITS", "row", "firstTForbit"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
