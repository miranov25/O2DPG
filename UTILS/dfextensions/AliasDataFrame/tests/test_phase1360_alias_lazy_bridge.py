"""Phase 13.60.ADF — alias <-> lazy-branch bridge.

T1  materialize over unloaded branch auto-loads + equals eager (A1)
T2  exact-load: only required branches loaded, >=3 decoys absent (A2)
T3  validate_aliases not broken; describe marks LAZY not BROKEN (A3)
T4  [must] genuine-missing still reported + registered-function alias case (A5)
T5  eager no-regression (A4)
T6  fails-without-fix guard (FM#12)

All tests call the public API the user hit the bug with.
"""
import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import uproot

from AliasDataFrame import AliasDataFrame

PI = 3.141592653589793


def _make_tree(path, n=2000):
    rng = np.random.default_rng(0)
    data = {
        "phiITSTPCAtVertex": rng.uniform(-PI, PI, n).astype("float32"),
        "tgl": rng.uniform(-1, 1, n).astype("float32"),
        "qpt": rng.uniform(-4, 4, n).astype("float32"),
        # decoys — must NOT be loaded by a sector/tgl materialize
        "decoy_a": rng.standard_normal(n).astype("float32"),
        "decoy_b": rng.standard_normal(n).astype("float32"),
        "decoy_c": rng.standard_normal(n).astype("float32"),
        "decoy_d": rng.standard_normal(n).astype("float32"),
    }
    with uproot.recreate(path) as f:
        f.mktree("t", {k: v.dtype for k, v in data.items()})
        f["t"].extend(data)
    return data


@pytest.fixture()
def tree_path():
    p = tempfile.mktemp(suffix=".root")
    _make_tree(p)
    yield p
    if os.path.exists(p):
        os.remove(p)


def _add_sector_aliases(adf):
    adf.add_alias("sector_bin180", "90*(phiITSTPCAtVertex/3.141592653589793)")
    adf.add_alias("tgl_bin10", "10*tgl")


@pytest.mark.invariance
def test_T1_materialize_over_unloaded_branch_equals_eager(tree_path):
    # lazy: aliases reference branches not yet loaded -> must auto-load, not NameError
    lz = AliasDataFrame.read_tree_lazy(tree_path, "t")
    _add_sector_aliases(lz)
    lz.materialize_aliases(names=["sector_bin180", "tgl_bin10"])
    assert "sector_bin180" in lz.df.columns
    assert "tgl_bin10" in lz.df.columns

    # eager reference
    eg = AliasDataFrame.read_tree_lazy(tree_path, "t")
    eg.ensure_branches(["phiITSTPCAtVertex", "tgl", "qpt", "decoy_a", "decoy_b", "decoy_c", "decoy_d"])
    _add_sector_aliases(eg)
    eg.materialize_aliases(names=["sector_bin180", "tgl_bin10"])
    assert np.allclose(lz.df["sector_bin180"].to_numpy(), eg.df["sector_bin180"].to_numpy(), equal_nan=True)
    assert np.allclose(lz.df["tgl_bin10"].to_numpy(), eg.df["tgl_bin10"].to_numpy(), equal_nan=True)


def test_T2_exact_load_decoys_not_loaded(tree_path):
    lz = AliasDataFrame.read_tree_lazy(tree_path, "t")
    _add_sector_aliases(lz)
    lz.materialize_aliases(names=["sector_bin180", "tgl_bin10"])
    cols = set(lz.df.columns)
    # required branches present
    assert {"phiITSTPCAtVertex", "tgl"} <= cols
    # >=3 decoys NOT loaded
    for decoy in ("decoy_a", "decoy_b", "decoy_c", "decoy_d", "qpt"):
        assert decoy not in cols, f"{decoy} should not have been loaded"


@pytest.mark.invariance
def test_T3_validate_and_describe_lazy_not_broken(tree_path):
    lz = AliasDataFrame.read_tree_lazy(tree_path, "t")
    _add_sector_aliases(lz)
    # validate: lazily-loadable aliases must NOT be reported broken
    broken = set(lz.validate_aliases())
    assert "sector_bin180" not in broken
    assert "tgl_bin10" not in broken
    # describe: marked LAZY, not BROKEN
    info = lz.describe_aliases(as_dict=True)
    assert info["sector_bin180"]["broken"] is False
    assert info["sector_bin180"]["lazy"] is True
    assert info["tgl_bin10"]["lazy"] is True


def test_T4_genuine_missing_still_reported_and_registered_fn(tree_path):  # [must]
    lz = AliasDataFrame.read_tree_lazy(tree_path, "t")
    # genuinely absent branch -> still broken, not silently masked
    lz.add_alias("ghost", "not_a_real_branch_zzz * 2")
    assert "ghost" in set(lz.validate_aliases())
    # registered-function alias over an available branch -> resolvable (not broken),
    # bounding the _parse_selection_columns / function-name handling
    lz.add_alias("abs_qpt", "abs(qpt)")
    assert "abs_qpt" not in set(lz.validate_aliases())
    lz.materialize_aliases(names=["abs_qpt"])
    assert np.allclose(lz.df["abs_qpt"].to_numpy(),
                       np.abs(lz.df["qpt"].to_numpy()), equal_nan=True)


def test_T5_eager_no_regression(tree_path):
    # eager frame: bridges must be inert (reader present but everything loaded == eager)
    eg = AliasDataFrame(pd.DataFrame({
        "phiITSTPCAtVertex": np.linspace(-PI, PI, 100, dtype="float32"),
        "tgl": np.linspace(-1, 1, 100, dtype="float32"),
    }))
    _add_sector_aliases(eg)
    # no lazy reader -> _lazy_available_names() empty -> validate identical to pre-change
    assert eg.validate_aliases() == []  # both resolvable from loaded columns
    eg.materialize_aliases(names=["sector_bin180"])
    assert np.allclose(eg.df["sector_bin180"].to_numpy(),
                       90 * (eg.df["phiITSTPCAtVertex"].to_numpy() / PI), equal_nan=True)
    info = eg.describe_aliases(as_dict=True)
    # eager + loaded after materialize -> not lazy
    assert info["sector_bin180"]["lazy"] is False


def test_T6_fails_without_fix_guard(tree_path):
    # Documents that the bridge is load-bearing: temporarily blind the reader, the
    # pre-fix behaviour (NameError on materialize) must reappear.
    lz = AliasDataFrame.read_tree_lazy(tree_path, "t")
    _add_sector_aliases(lz)
    saved = lz._lazy_autoload_set
    lz._lazy_autoload_set = lambda names: set()  # simulate the unbridged path
    try:
        with pytest.raises(Exception):
            lz.materialize_aliases(names=["sector_bin180"])
    finally:
        lz._lazy_autoload_set = saved
    # with the bridge restored it succeeds
    lz.materialize_aliases(names=["sector_bin180"])
    assert "sector_bin180" in lz.df.columns


# ── Invariance extensions (C-0 gate): lazy == eager across alias shapes ──

def _lazy_with_shapes(path):
    a = AliasDataFrame.read_tree_lazy(path, "t")
    a.add_alias("sector", "18*(phiITSTPCAtVertex/3.141592653589793)")  # over a branch
    a.add_alias("dsector", "sector-int(sector)")                        # CHAINED: over an alias
    a.add_alias("qpt_bin5", "5*qpt", dtype="int8")                      # dtype-bearing
    return a


@pytest.mark.invariance
def test_T7_chained_alias_transitive_autoload_equals_eager(tree_path):
    # alias-on-alias: auto-load must resolve the transitive branch dependency (phi via sector)
    lz = _lazy_with_shapes(tree_path)
    lz.materialize_aliases(names=["dsector"], with_dependencies=True)
    assert "phiITSTPCAtVertex" in lz.df.columns  # transitive branch was loaded
    eg = _lazy_with_shapes(tree_path)
    eg.ensure_branches(["phiITSTPCAtVertex", "qpt", "tgl"])
    eg.materialize_aliases(names=["dsector"], with_dependencies=True)
    assert np.allclose(lz.df["dsector"].to_numpy(), eg.df["dsector"].to_numpy(), equal_nan=True)


@pytest.mark.invariance
def test_T8_dtype_bearing_alias_lazy_equals_eager(tree_path):
    # the int8/float16 compression dtypes must be identical lazy vs eager
    lz = _lazy_with_shapes(tree_path)
    lz.materialize_aliases(names=["qpt_bin5"])
    eg = _lazy_with_shapes(tree_path)
    eg.ensure_branches(["qpt"])
    eg.materialize_aliases(names=["qpt_bin5"])
    assert str(lz.df["qpt_bin5"].dtype) == str(eg.df["qpt_bin5"].dtype) == "int8"
    assert np.array_equal(lz.df["qpt_bin5"].to_numpy(), eg.df["qpt_bin5"].to_numpy())


def test_T9_pattern_selection_on_lazy(tree_path):
    # materialize_aliases(pattern=...) — the exact form the user hit (".*sector.*")
    lz = _lazy_with_shapes(tree_path)
    lz.materialize_aliases(pattern=".*sector.*", with_dependencies=True)
    assert "sector" in lz.df.columns and "dsector" in lz.df.columns
