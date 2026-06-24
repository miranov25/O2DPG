"""PHASE_13_62_ADF Stage 1 — lazy-branch reconciliation + missing-data diagnostics.

S1-D1: ensure_branches requests only real, not-yet-present TTree branches.
S1-D2: anything unresolvable is classified — genuine missing input raises a cause-naming
       BranchNotFoundError; names resolved elsewhere (aliases, subframe columns) are skipped.

FM#12: each test calls the public API the user hit. T2 (missing-data error) and T6 (typo
negative control) are the critical pair; both fail on pre-fix code.
"""
import numpy as np
import pandas as pd
import pytest
import uproot

from dfextensions.AliasDataFrame import AliasDataFrame
from dfextensions.AliasDataFrame.AliasDataFrame import BranchNotFoundError

N = 2000
RAW = ["ncl", "dcar_itstpc", "hasITSTPC", "phiITSTPCAtVertex"]
DECOYS = ["decoy_a", "decoy_b", "decoy_c", "decoy_d"]


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


def test_T1_hand_added_column_readable(root_path):
    """Face 1: a hand-added in-memory column is not re-requested from the tree."""
    adf = _lazy(root_path)
    adf.df["rowmask_3_5"] = np.arange(N, dtype=np.uint32)
    adf.ensure_branches(["rowmask_3_5", "ncl"])      # must NOT raise
    assert "rowmask_3_5" in adf.df.columns and "ncl" in adf.df.columns


def test_T2_genuine_missing_raises_cause_naming_error(root_path):
    """Headline: a genuinely absent name raises BranchNotFoundError naming the cause."""
    adf = _lazy(root_path)
    with pytest.raises(BranchNotFoundError) as ei:
        adf.ensure_branches(["ncl", "totally_bogus_xyz"])
    assert isinstance(ei.value.missing, set)            # .missing is a set, not a string
    assert "totally_bogus_xyz" in ei.value.missing
    msg = str(getattr(ei.value, "message", "") or ei.value)
    assert ("typo" in msg) or ("missing input" in msg)


def test_T3_alias_and_subframe_columns_not_raised(root_path):
    """Names resolved elsewhere (alias, subframe column) are skipped, not raised."""
    adf = _lazy(root_path)
    adf.ensure_columns("phiITSTPCAtVertex")
    adf.add_alias("myAlias", "ncl*2")
    coeffs = AliasDataFrame(pd.DataFrame({"k": [0, 1, 2], "sfcol": [1.0, 2.0, 3.0]}))
    adf.register_subframe("SF", coeffs, index_columns=["k"])
    adf.ensure_branches(["myAlias", "sfcol", "phiITSTPCAtVertex"])   # must NOT raise


def test_T4_exact_load(root_path):
    """Only the referenced real branches load; >=3 decoys stay unloaded."""
    adf = _lazy(root_path)
    before = set(adf.df.columns)
    adf.ensure_branches(["ncl", "dcar_itstpc", "hasITSTPC"])
    newly = set(adf.df.columns) - before
    assert newly == {"ncl", "dcar_itstpc", "hasITSTPC"}
    assert all(d not in adf.df.columns for d in DECOYS)


def test_T5_eager_unchanged(root_path):
    """Eager ADF: present column no-op; genuinely missing still raises."""
    eager = AliasDataFrame(pd.DataFrame({"ncl": [1, 2, 3]}))
    cols = list(eager.df.columns)
    eager.ensure_branches(["ncl"])
    assert list(eager.df.columns) == cols
    with pytest.raises(BranchNotFoundError):
        eager.ensure_branches(["nope"])


def test_T6_typo_negative_control(root_path):
    """Direct ensure_branches on a pure typo still raises (no blanket silent-skip)."""
    adf = _lazy(root_path)
    with pytest.raises(BranchNotFoundError):
        adf.ensure_branches(["definitely_not_a_branch"])


def test_C1_subframe_alias_lazy_draw_end_to_end(tmp_path):
    """C1 (reviewer-requested end-to-end): lazy ADF + register_subframe + an alias that
    references a subframe column + draw() -> succeeds AND computes correct values via the
    on-demand subframe merge (not just 'name does not raise'). Uses plain A.col syntax.
    """
    rng = np.random.default_rng(1)
    nrow = 4000
    kb = rng.integers(0, 8, nrow)
    p = tmp_path / "t.root"
    with uproot.recreate(str(p)) as f:
        f["t"] = {"x": rng.normal(0, 1, nrow).astype(np.float32),
                  "kbin": kb.astype(np.int32),
                  "yobs": rng.normal(0, 1, nrow).astype(np.float32)}
    adf = AliasDataFrame.read_tree_lazy(str(p), "t")
    adf.draw_lazy = True
    adf.ensure_columns("kbin")                      # subframe index available (as production does)
    coeff = AliasDataFrame(pd.DataFrame({"kbin": np.arange(8, dtype=np.int32),
                                         "slope": np.linspace(1.0, 8.0, 8)}))
    adf.register_subframe("FIT", coeff, index_columns=["kbin"])
    adf.add_alias("pred", "FIT.slope * x")          # plain A.col (no backticks)
    adf.add_alias("resid", "yobs - pred")
    adf.draw("resid:x", type="profile", bins=20, min_entries=5)   # lazy draw of subframe-backed alias
    adf.materialize_aliases(names=["resid"])
    slopes = np.linspace(1.0, 8.0, 8)
    got = adf.df["resid"].to_numpy()
    exp = adf.df["yobs"].to_numpy() - slopes[adf.df["kbin"].to_numpy()] * adf.df["x"].to_numpy()
    assert np.nanmax(np.abs(got - exp)) < 1e-4      # correct values, not just no-raise
