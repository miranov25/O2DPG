"""PHASE_13_72_ADF - bug-fix mini-phase.

Bug A: subframe-column prefix collision (boundary-blind substring rewrite).
Bug B: scalar/string-dtype cast crash + eval's error-swallowing misdirection.
Every fix test is a protocol repro: fails before the fix, passes after (FM#12).
"""
import numpy as np
import pandas as pd
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AliasDataFrame import AliasDataFrame

try:
    import uproot  # noqa: F401
    _HAS_UPROOT = True
except Exception:
    _HAS_UPROOT = False


def _parent_with_sub(cols):
    """Parent joined by 'run' to a subframe S carrying the given {name: values} columns."""
    n = 5
    parent = AliasDataFrame(pd.DataFrame({"run": [0, 1, 2, 0, 1], "x": np.arange(n) * 1.0}))
    sub = AliasDataFrame(pd.DataFrame({"run": [0, 1, 2], **cols}))
    parent.register_subframe("S", sub, index_columns=["run"])
    return parent


# ============================ Bug A ============================
def test_A1_prefix_pair_dotted_chain():
    """A2 repro: S.stepZ14 / S.stepZ14pt in one expression -> correct substitution + values."""
    p = _parent_with_sub({"stepZ14": [10., 20., 30.], "stepZ14pt": [100., 200., 300.]})
    p.add_alias("both", "S.stepZ14 + S.stepZ14pt", dtype="float32")
    p.materialize_aliases(names=["both"])
    ref = np.array([110, 220, 330, 110, 220], dtype="float32")   # run-joined stepZ14+stepZ14pt
    assert np.allclose(p.df["both"].values, ref)


def test_A2_forced_silent_wrong_values_P0():
    """P0 pin: with a prefix-colliding pair in one expression AND poison parent columns of
    the same leaf names, the alias must compute NUMERICALLY CORRECT subframe-joined values.
    Asymmetric magnitudes make any mangling/mis-resolution a detectably wrong number.
    Before the fix the shorter prefix mangled the longer -> silently wrong values."""
    p = _parent_with_sub({"stepZ14": [1., 2., 3.], "stepZ14pt": [1000., 2000., 3000.]})
    p.df["stepZ14"] = -999.0            # poison: bare parent columns of the same leaf names
    p.df["stepZ14pt"] = -999.0
    p.add_alias("v", "S.stepZ14 + S.stepZ14pt", dtype="float32")
    p.materialize_aliases(names=["v"])
    ref = np.array([1001, 2002, 3003, 1001, 2002], dtype="float32")   # subframe-joined truth
    got = p.df["v"].values
    assert np.allclose(got, ref), f"silently-wrong: {got.tolist()} != {ref.tolist()}"


@pytest.mark.invariance
def test_A3_order_insensitive():
    """Both textual orders of a prefix-colliding pair give identical correct results."""
    cols = {"stepZ14": [10., 20., 30.], "stepZ14pt": [100., 200., 300.]}
    p1 = _parent_with_sub(cols); p1.add_alias("r", "S.stepZ14 + S.stepZ14pt", dtype="float32")
    p2 = _parent_with_sub(cols); p2.add_alias("r", "S.stepZ14pt + S.stepZ14", dtype="float32")
    p1.materialize_aliases(names=["r"]); p2.materialize_aliases(names=["r"])
    assert np.allclose(p1.df["r"].values, p2.df["r"].values)
    assert np.allclose(p1.df["r"].values, [110, 220, 330, 110, 220])


def test_A4_negative_control_no_false_rewrite():
    """Non-colliding references are unaffected; a lone reference still resolves."""
    p = _parent_with_sub({"stepZ14": [10., 20., 30.], "other": [7., 8., 9.]})
    p.add_alias("lone", "S.stepZ14", dtype="float32")
    p.add_alias("mix", "S.stepZ14 + S.other", dtype="float32")
    p.materialize_aliases(names=["lone", "mix"])
    assert np.allclose(p.df["lone"].values, [10, 20, 30, 10, 20])
    assert np.allclose(p.df["mix"].values, [17, 28, 39, 17, 28])


def test_A5_two_simultaneous_prefix_pairs():
    """Sonnet18's production shape: TWO prefix pairs in ONE expression, progressive-loop
    mutation must not cross-contaminate — all four substitutions correct in one pass."""
    p = _parent_with_sub({
        "stepZ14": [10., 20., 30.], "stepZ14pt": [100., 200., 300.],
        "aZ14": [1., 2., 3.], "aZ14pt": [4., 5., 6.],
    })
    p.add_alias("q", "S.stepZ14 + S.stepZ14pt + S.aZ14 + S.aZ14pt", dtype="float32")
    p.materialize_aliases(names=["q"])
    ref = np.array([10+100+1+4, 20+200+2+5, 30+300+3+6] * 1, dtype="float32")
    ref = np.array([115, 227, 339, 115, 227], dtype="float32")
    assert np.allclose(p.df["q"].values, ref)


# ============================ Bug B ============================
@pytest.mark.parametrize("d", ["float16", np.dtype("float16"), np.float16, None])
def test_B1_scalar_string_dtype_materializes(d):
    """Zero-dependency constant alias with scalar result + (string|dtype|type|None) dtype:
    materializes with the correct value AND the correct column dtype - not merely 'no error'."""
    a = AliasDataFrame(pd.DataFrame({"x": np.arange(5) * 1.0}))
    a.add_alias("c", "1.0", dtype=d)
    a.materialize_aliases(names=["c"])
    col = a.df["c"]
    assert np.allclose(col.values, 1.0)
    if d is not None:
        assert col.dtype == np.dtype(d)                       # exact requested dtype
        assert col.iloc[0] == np.dtype(d).type(1.0)           # independent cast reference


def test_B2_no_error_swallow_true_type_surfaces():
    """With the eval swallow removed, a genuine internal error surfaces as its TRUE type at
    its true site - not a misleading NameError from Step 3."""
    a = AliasDataFrame(pd.DataFrame({"x": np.arange(5) * 1.0}))

    def boom(v):
        raise RuntimeError("intentional")

    a.register_function("boom", boom)
    a.add_alias("bad", "boom(x)")
    with pytest.raises(RuntimeError, match="intentional"):
        a.eval("bad + 1")


@pytest.mark.invariance
@pytest.mark.skipif(not _HAS_UPROOT, reason="uproot required for lazy/chain legs")
def test_B3_cross_path_parity(tmp_path):
    """Cross-path parity (Inv, cf. I2_6): the Bug-B constant alias yields the same value on
    eager, lazy, and chain frames."""
    df = pd.DataFrame({"x": np.arange(6) * 1.0})
    f0 = str(tmp_path / "a.root"); AliasDataFrame(df).export_tree(f0, treename="tree")
    f1 = str(tmp_path / "b.root"); AliasDataFrame(df).export_tree(f1, treename="tree")

    eager = AliasDataFrame(df.copy())
    lazy = AliasDataFrame.read_tree_lazy(f0, tree_name="tree")
    chain = AliasDataFrame.read_chain_lazy([f0 + ":tree", f1 + ":tree"])
    vals = []
    for adf in (eager, lazy, chain):
        adf.add_alias("c", "1.0", dtype="float16")
        adf.materialize_aliases(names=["c"])
        vals.append(float(adf.df["c"].iloc[0]))
    assert vals[0] == vals[1] == vals[2] == 1.0


def test_REG_normal_subframe_join_unaffected():
    """D-A touches the shared join path; a plain non-colliding join must be unchanged."""
    p = _parent_with_sub({"pt": [1., 2., 3.]})
    p.add_alias("j", "S.pt * 2", dtype="float32")
    p.materialize_aliases(names=["j"])
    assert np.allclose(p.df["j"].values, [2, 4, 6, 2, 4])
