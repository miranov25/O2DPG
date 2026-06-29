"""
PHASE_13_62_ADF Stage 2a (Fix A) — adf[col] = value write-through.

Fix A replaced the previous TypeError block in AliasDataFrame.__setitem__ with a
write-through that also syncs the lazy reader's ``loaded_branches`` bookkeeping, so a
hand-added column is immediately present and is never re-requested from the TTree.

All tests drive the same public API the bug used — ``adf[col] = value`` — and fail
against the pre-Fix-A source (which raised ``TypeError``), per FM#12.

Also folds in the P3-1 advisory from the Stage 1 CRR review: a leaked bare-leaf
subframe-column name reaching ``ensure_branches`` (the backtick-reference shape) is
classified ``misrouted`` and does not raise, while a genuine typo does.

Value-shape coverage note (reviewer carry-in): the supported shapes are those pandas
accepts for ``df[key] = value`` — numpy array, index-aligned Series, list, scalar.
Awkward arrays are NOT auto-converted; convert explicitly (``ak.to_numpy`` for regular,
``.tolist()`` for ragged). The sandbox runs pandas 3.x; on the production stack
(pandas 1.5.3) direct awkward assignment must not be relied upon — hence the explicit
``ak.to_numpy`` path is the supported one and is what this suite asserts.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest
import uproot

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AliasDataFrame import AliasDataFrame  # noqa: E402

try:
    from exceptions import BranchNotFoundError
except ImportError:  # pragma: no cover
    from AliasDataFrame import BranchNotFoundError  # type: ignore


def _lazy_adf(tmp_path, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    p = tmp_path / "fixa.root"
    with uproot.recreate(str(p)) as f:
        f["t"] = {"x": rng.normal(0, 1, n).astype(np.float32),
                  "kbin": rng.integers(0, 8, n).astype(np.int32)}
    adf = AliasDataFrame.read_tree_lazy(str(p), "t")
    adf.draw_lazy = True
    return adf


# --------------------------------------------------------------------------- #
# Eager write-through (value shapes)
# --------------------------------------------------------------------------- #

def test_S2A_1_eager_numpy_writethrough():
    adf = AliasDataFrame(pd.DataFrame({"x": np.arange(5.0)}))
    adf["new"] = np.arange(5.0) * 2
    assert "new" in adf.df.columns
    assert np.allclose(adf.df["new"], np.arange(5.0) * 2)
    assert np.allclose(adf["new"], np.arange(5.0) * 2)   # read back via adf[]


def test_S2A_2_eager_series_index_aligned():
    adf = AliasDataFrame(pd.DataFrame({"x": np.arange(5.0)}))
    adf["new"] = pd.Series([5, 4, 3, 2, 1], index=adf.df.index)
    assert list(adf.df["new"]) == [5, 4, 3, 2, 1]


def test_S2A_3_eager_scalar_broadcast():
    adf = AliasDataFrame(pd.DataFrame({"x": np.arange(5.0)}))
    adf["new"] = 7
    assert (adf.df["new"] == 7).all()


def test_S2A_4_non_string_key_raises():
    adf = AliasDataFrame(pd.DataFrame({"x": np.arange(5.0)}))
    with pytest.raises(TypeError):
        adf[("bad",)] = np.arange(5.0)


def test_S2A_5_aliases_mapping_still_immutable():
    """Fix A touches column assignment only; the _ReadOnlyAliasDict guard on
    adf.aliases (the DO-NOT-TOUCH :827 method) is unaffected."""
    adf = AliasDataFrame(pd.DataFrame({"x": np.arange(5.0)}))
    with pytest.raises(TypeError):
        adf.aliases["foo"] = "x + 1"


# --------------------------------------------------------------------------- #
# Lazy write-through + bookkeeping sync
# --------------------------------------------------------------------------- #

def test_S2A_6_lazy_writethrough_and_loaded_branches_sync(tmp_path):
    adf = _lazy_adf(tmp_path)
    adf.ensure_columns("x")
    mask = (adf.df["x"].to_numpy() > 0).astype(np.uint8)
    adf["rowmask"] = mask                                  # Fix A on a lazy ADF
    assert "rowmask" in adf.df.columns
    assert np.array_equal(adf.df["rowmask"].to_numpy(), mask)
    # bookkeeping: the column is recorded as loaded so it is not re-requested
    assert "rowmask" in adf._lazy_reader.loaded_branches


def test_S2A_7_lazy_handadded_readable_no_rerequest(tmp_path):
    """After write-through, ensure_columns on the hand-added name is a no-op
    (Stage 1 reconciliation) and the column is usable in eval."""
    adf = _lazy_adf(tmp_path)
    adf.ensure_columns("x")
    mask = (adf.df["x"].to_numpy() > 0).astype(np.uint8)
    adf["rowmask"] = mask
    adf.ensure_columns("rowmask")                          # must NOT raise
    assert int(adf.df.eval("rowmask > 0").sum()) == int(mask.sum())


def test_S2A_8_lazy_length_mismatch_raises_and_not_registered(tmp_path):
    """A bad value shape raises (from pandas) and the name is NOT recorded as
    loaded — the write happens before the bookkeeping."""
    adf = _lazy_adf(tmp_path)
    adf.ensure_columns("x")
    with pytest.raises(Exception):
        adf["bad"] = np.arange(3.0)                        # wrong length
    assert "bad" not in adf._lazy_reader.loaded_branches
    assert "bad" not in adf.df.columns


# --------------------------------------------------------------------------- #
# Awkward conversion path (make_row_group_mask motivating case)
# --------------------------------------------------------------------------- #

def test_S2A_9_awkward_regular_via_to_numpy():
    ak = pytest.importorskip("awkward")
    adf = AliasDataFrame(pd.DataFrame({"x": np.arange(5.0)}))
    adf["from_ak"] = ak.to_numpy(ak.Array(np.arange(5.0)))   # supported: explicit conversion
    assert np.allclose(adf.df["from_ak"], np.arange(5.0))


# --------------------------------------------------------------------------- #
# P3-1 (Stage 1 CRR advisory): backtick-leak bare-leaf graceful degrade
# --------------------------------------------------------------------------- #

def test_P3_1_backtick_leak_bare_leaf_not_raised(tmp_path):
    """A backtick subframe reference leaks the bare leaf (e.g. 'slope'); when that
    name reaches ensure_branches it must classify as a subframe column (misrouted)
    and not raise, while a genuine typo still raises BranchNotFoundError."""
    adf = _lazy_adf(tmp_path)
    adf.ensure_columns("kbin")
    coeff = AliasDataFrame(pd.DataFrame({"kbin": np.arange(8, dtype=np.int32),
                                         "slope": np.linspace(1, 8, 8)}))
    adf.register_subframe("FIT", coeff, index_columns=["kbin"])
    adf.ensure_branches(["slope", "x"])                    # bare-leaf leak shape -> no raise
    with pytest.raises(BranchNotFoundError):
        adf.ensure_branches(["nonexistent_zzz"])          # genuine typo still raises
