"""PHASE_13_67_ADF — public-API chain metadata recovery tests (CRR revision).

Addresses the main-reviewer P0-1/P1-2/P1-4 gaps: the earlier suite exercised the internal
helpers (_normalize_chain_meta / _check_chain_metadata_compatibility / _apply_recovered_
metadata) directly, but never read a metadata-bearing MULTI-FILE chain through the actual
public method read_chain_lazy. These tests do, using REAL files written by export_tree
(F-5: real fixtures, never dict-mocked). uproot-gated -> run on alma2, skip in a ROOT-free
sandbox. The fail-loud test (P1-4) is pure Python and runs everywhere.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AliasDataFrame import AliasDataFrame          # noqa: E402
from exceptions import ChainMetadataCompatibilityError  # noqa: E402

TREE = "tree"


def _write_file(path, n, alias_expr, seed):
    """Write ONE metadata-bearing ROOT file via export_tree: branches x,y,z + alias 'd'."""
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        "x": rng.randn(n).astype(np.float32),
        "y": (rng.randn(n).astype(np.float32) + 5.0),   # avoid divide-by-zero in x/y
        "z": (rng.randn(n).astype(np.float32) + 5.0),
    })
    adf = AliasDataFrame(df)
    adf.add_alias("d", alias_expr)
    adf.export_tree(str(path), treename=TREE)
    return str(path)


@pytest.fixture
def compatible_meta_chain(tmp_path):
    """Two files, SAME alias definition d=x/y -> compatible metadata across the chain."""
    pytest.importorskip("uproot")
    f0 = _write_file(tmp_path / "c0.root", 1000, "x/y", seed=1)
    f1 = _write_file(tmp_path / "c1.root", 1500, "x/y", seed=2)
    return [f0, f1]


@pytest.fixture
def mismatched_meta_chain(tmp_path):
    """Two files, DIFFERENT alias definition for 'd' -> incompatible metadata."""
    pytest.importorskip("uproot")
    f0 = _write_file(tmp_path / "m0.root", 1000, "x/y", seed=3)
    f1 = _write_file(tmp_path / "m1.root", 1000, "x/z", seed=4)   # d differs
    return [f0, f1]


# ── P0-1: the headline claim, through the PUBLIC chain API ──
@pytest.mark.invariance
def test_read_chain_lazy_recovers_alias_public_api(compatible_meta_chain):
    """read_chain_lazy over a real metadata-bearing chain restores the alias, and the
    restored alias evaluates correctly through the public path (auto-loading x,y)."""
    adf = AliasDataFrame.read_chain_lazy(compatible_meta_chain, tree_name=TREE)
    assert "d" in adf.aliases, "alias 'd' not recovered from chain UserInfo"
    # evaluate the restored alias end-to-end; d == x/y on the concatenated chain
    got = np.asarray(adf.eval("d"), dtype=float)
    expect = np.asarray(adf.eval("x"), dtype=float) / np.asarray(adf.eval("y"), dtype=float)
    assert got.shape[0] == 2500                       # 1000 + 1500 rows
    assert np.allclose(got, expect, equal_nan=True)


# ── P1-2: metadata_conflict policy, all three modes, on a REAL mismatch ──
@pytest.mark.invariance
def test_metadata_conflict_error_raises_with_offswitch(mismatched_meta_chain):
    with pytest.raises(ChainMetadataCompatibilityError) as exc:
        AliasDataFrame.read_chain_lazy(mismatched_meta_chain, tree_name=TREE,
                                       validate_branches="strict")  # default metadata_conflict='error'
    msg = str(exc.value)
    assert "metadata_conflict='skip'" in msg or "metadata_conflict='warn'" in msg, \
        "error must name the off-switch"


def test_metadata_conflict_warn_proceeds(mismatched_meta_chain):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        adf = AliasDataFrame.read_chain_lazy(mismatched_meta_chain, tree_name=TREE,
                                             validate_branches="strict", metadata_conflict="warn")
    assert adf is not None
    assert any("metadata" in str(x.message).lower() for x in w), "warn mode must emit a warning"


def test_metadata_conflict_skip_proceeds_silently(mismatched_meta_chain):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        adf = AliasDataFrame.read_chain_lazy(mismatched_meta_chain, tree_name=TREE,
                                             validate_branches="strict", metadata_conflict="skip")
    assert adf is not None
    assert not any("differ" in str(x.message).lower() for x in w), "skip must not warn about the conflict"


# ── Decision 2: union/intersection + metadata -> error by default ──
@pytest.mark.invariance
def test_union_with_metadata_default_errors(compatible_meta_chain):
    with pytest.raises(ChainMetadataCompatibilityError):
        AliasDataFrame.read_chain_lazy(compatible_meta_chain, tree_name=TREE,
                                       validate_branches="union")   # default metadata_conflict='error'
    # ...and the off-switch works:
    adf = AliasDataFrame.read_chain_lazy(compatible_meta_chain, tree_name=TREE,
                                         validate_branches="union", metadata_conflict="skip")
    assert adf is not None


# ── P1-4: application failures FAIL LOUD (pure Python; runs everywhere) ──
def test_apply_recovered_metadata_warns_on_bad_dtype():
    adf = AliasDataFrame(pd.DataFrame({"x": [1.0, 2.0]}))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        adf._apply_recovered_metadata(
            {"aliases": {}, "dtypes": {"x": "not_a_real_dtype_zzz"}, "compression": {}})
    assert any("_apply_recovered_metadata" in str(x.message) for x in w), \
        "a bad dtype must warn, not be silently swallowed"


def test_apply_recovered_metadata_bad_alias_does_not_abort_rest():
    adf = AliasDataFrame(pd.DataFrame({"x": [1.0, 2.0]}))
    orig = adf.add_alias

    def boom(n, e, *a, **k):
        if n == "bad":
            raise ValueError("simulated bad alias")
        return orig(n, e, *a, **k)

    adf.add_alias = boom
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        adf._apply_recovered_metadata(
            {"aliases": {"bad": "x+", "good": "x*2"}, "dtypes": {}, "compression": {}})
    assert any("_apply_recovered_metadata" in str(x.message) for x in w)
    assert "good" in adf.aliases, "one bad alias must not abort applying the rest"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
