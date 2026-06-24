"""BUG_20260624 — ensure_columns(): lazy-branch bridge for direct-access paths.

FM#12: each test calls the same public API the user hit (adf.df.eval / adf.df[col])
on a lazy ADF, asserts the pre-fix failure mode, then asserts ensure_columns fixes it.
Exact-load discipline: >=3 decoy branches present in the tree must NOT be loaded.
"""
import numpy as np
import pandas as pd
import pytest
import uproot

from dfextensions.AliasDataFrame import AliasDataFrame

N = 2000
RAW = ["ncl", "dcar_itstpc", "hasITSTPC", "nClITS", "phiITSTPCAtVertex"]
DECOYS = ["decoy_a", "decoy_b", "decoy_c", "decoy_d"]   # >=3 decoys
SEL = "(ncl>50)&(abs(dcar_itstpc)<0.1)&(hasITSTPC>0)&(nClITS>4)"


@pytest.fixture
def root_path(tmp_path):
    rng = np.random.default_rng(0)
    data = {b: rng.normal(60, 20, N).astype(np.float32) for b in RAW}
    for d in DECOYS:
        data[d] = rng.normal(0, 1, N).astype(np.float32)
    p = tmp_path / "tracks.root"
    with uproot.recreate(str(p)) as f:
        f["tree"] = data
    return str(p)


def _lazy(root_path):
    adf = AliasDataFrame.read_tree_lazy(root_path, "tree")
    adf.draw_lazy = True
    return adf


def test_T1_eval_fails_before_succeeds_after(root_path):
    """A1 (FM#12): df.eval(selection) raises before, succeeds after; exact-load (A2)."""
    adf = _lazy(root_path)
    before = set(adf.df.columns)
    with pytest.raises(Exception):           # UndefinedVariableError pre-fix
        adf.df.eval(SEL)
    adf.ensure_columns(SEL)
    mask = adf.df.eval(SEL)                   # now succeeds
    assert mask.dtype == bool and len(mask) == N
    newly = set(adf.df.columns) - before
    assert newly == {"ncl", "dcar_itstpc", "hasITSTPC", "nClITS"}  # exactly the referenced
    assert all(d not in adf.df.columns for d in DECOYS)            # no decoys


def test_T2_mixed_args(root_path):
    """A4: expression string + list + bare name all resolve; unrelated decoys absent."""
    adf = _lazy(root_path)
    adf.ensure_columns("(ncl>50)", ["dcar_itstpc", "hasITSTPC"], "nClITS")
    for c in ("ncl", "dcar_itstpc", "hasITSTPC", "nClITS"):
        assert c in adf.df.columns
    assert all(d not in adf.df.columns for d in DECOYS)


def test_T3_eager_noop():
    """A3: on an eager ADF ensure_columns is a no-op (frame unchanged)."""
    eager = AliasDataFrame(pd.DataFrame({"ncl": [60, 70, 80]}))
    cols = list(eager.df.columns)
    eager.ensure_columns(SEL)
    assert list(eager.df.columns) == cols


def test_T4_math_constants_not_loaded(root_path):
    """A-const: 'pi' is a math constant, must not be requested as a branch."""
    adf = _lazy(root_path)
    adf.ensure_columns("abs(phiITSTPCAtVertex/pi*180)")  # must not raise BranchNotFound
    assert "phiITSTPCAtVertex" in adf.df.columns
    assert "pi" not in adf.df.columns
