"""PHASE_13_73_ADF - source-scoped alias resolution (add_alias/add_aliases source=).

AST-based binding of bare fit formulas to a registered subframe. Tests T-1..T-11 of
proposal Rev 1.1. FM#12: every fix test fails without the feature.
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

N = 12


def _parent(**extra):
    df = pd.DataFrame({"run": np.arange(N) % 3, "qpt": np.arange(N) * 1.0,
                       "tgl": np.arange(N) * 0.5, **extra})
    return AliasDataFrame(df)


def _fit(cols, index=("run",)):
    return AliasDataFrame(pd.DataFrame({"run": [0, 1, 2], **cols}))


# ---- T-1: resolution correctness (Call args + Attribute passthrough) ----
def test_T1_resolution_call_args_and_attribute_passthrough():
    p = _parent()
    p.register_subframe("FIT", _fit({"p0": [1., 2., 3.], "p1": [4., 5., 6.]}), index_columns=["run"])
    p.add_alias("nl", "sqrt(p0**2 + (p1*qpt)**2)", source="FIT")
    assert p.aliases["nl"] == "sqrt(FIT.p0**2 + (FIT.p1*qpt)**2)"   # sqrt untouched; args resolved
    p.add_alias("already", "FIT.p0 + qpt", source="FIT")
    assert p.aliases["already"] == "FIT.p0 + qpt"                    # Attribute never touched


# ---- T-2 [F-1, CRITICAL]: registration against an unloaded-but-available lazy branch ----
@pytest.mark.skipif(not _HAS_UPROOT, reason="uproot required for lazy reader")
def test_T2_lazy_unloaded_branch_resolves(tmp_path):
    """The flagship workflow: on a lazy ADF the parent variable (qpt) is an available-but-NOT-
    loaded branch at registration. It must resolve as parent (bare), not raise R2."""
    f = str(tmp_path / "m.root")
    _parent().export_tree(f, treename="tree")
    lz = AliasDataFrame.read_tree_lazy(f, tree_name="tree")
    lz.register_subframe("FIT", _fit({"intercept": [1., 2., 3.], "slope_qpt": [10., 20., 30.]}),
                         index_columns=["run"])
    assert list(lz.df.columns) == []                                  # nothing loaded yet
    lz.add_alias("corr", "intercept + slope_qpt*qpt", source="FIT")   # must NOT raise
    assert lz.aliases["corr"] == "FIT.intercept + FIT.slope_qpt*qpt"
    lz.ensure_columns(["qpt", "run"])
    g = np.arange(N) % 3
    ref = np.array([1., 2., 3.])[g] + np.array([10., 20., 30.])[g] * (np.arange(N) * 1.0)
    assert np.allclose(lz.eval("corr").values, ref)


# ---- T-3: R1 shadowing raises loudly, naming both frames ----
def test_T3_R1_shadow_raises():
    p = _parent(time_s=np.arange(N) * 1.0)
    p.register_subframe("FIT", _fit({"time_s": [9., 9., 9.], "p0": [1., 2., 3.]}),
                        index_columns=["run"])
    with pytest.raises(ValueError, match="time_s"):
        p.add_alias("bad", "p0*time_s", source="FIT")


# ---- T-4 [Inv]: R1a exemption by MEMBERSHIP in index_columns, not name coincidence ----
@pytest.mark.invariance
def test_T4_R1a_index_exemption_by_membership():
    p = _parent()
    # 'run' is the index column -> exempt from R1, resolves to parent (stays bare)
    p.register_subframe("FIT", _fit({"p0": [1., 2., 3.]}), index_columns=["run"])
    p.add_alias("a1", "p0 + run", source="FIT")
    assert p.aliases["a1"] == "FIT.p0 + run"
    # Control: an identically-named column that is NOT an index column DOES shadow-raise.
    q = _parent(p9=np.arange(N) * 1.0)
    q.register_subframe("G", _fit({"p9": [7., 7., 7.], "p0": [1., 2., 3.]}), index_columns=["run"])
    with pytest.raises(ValueError, match="p9"):
        q.add_alias("bad", "p0 + p9", source="G")


# ---- T-5: R2 unknown token raises at registration, naming token + formula ----
def test_T5_R2_unknown_token_raises():
    p = _parent()
    p.register_subframe("FIT", _fit({"p0": [1., 2., 3.]}), index_columns=["run"])
    with pytest.raises(ValueError, match="nosuchthing"):
        p.add_alias("bad", "p0 + nosuchthing", source="FIT")


# ---- T-6 [Inv]: backward compatibility - source=None is bit-identical ----
@pytest.mark.invariance
def test_T6_backward_compat_source_none():
    p = _parent()
    p.register_subframe("FIT", _fit({"p0": [1., 2., 3.]}), index_columns=["run"])
    p.add_alias("y", "qpt*2")                       # no source= at all
    p.add_alias("z", "qpt*3", source=None)          # explicit None
    assert p.aliases["y"] == "qpt*2" and p.aliases["z"] == "qpt*3"   # untouched
    p.materialize_aliases(names=["y", "z"])
    assert np.allclose(p.df["y"].values, np.arange(N) * 2.0)


# ---- T-7: prefix-collision regression (the 13.72 bug class, via the NEW entry point) ----
def test_T7_prefix_collision_structurally_impossible():
    p = _parent()
    p.register_subframe("F", _fit({"stepZ14": [1., 2., 3.], "stepZ14pt": [10., 20., 30.]}),
                        index_columns=["run"])
    p.add_alias("pc", "stepZ14 + stepZ14pt", source="F")
    assert p.aliases["pc"] == "F.stepZ14 + F.stepZ14pt"     # NOT 'F.stepZ14 + F.stepZ14pt' mangled
    p.materialize_aliases(names=["pc"])
    g = np.arange(N) % 3
    ref = np.array([1., 2., 3.])[g] + np.array([10., 20., 30.])[g]
    assert np.allclose(p.df["pc"].values, ref)


# ---- T-8: PHASE_13_73_FIX - add_aliases removed; a loop is the supported batch form ----
def test_T8_batch_binding_via_loop_with_per_alias_dtype():
    """PHASE_13_73_FIX_ADF (D-1): add_aliases was removed. The supported way to bind a whole
    fit's formulas is a loop over add_alias(..., source=...) - which is strictly MORE
    expressive, because it takes a PER-ALIAS dtype (add_aliases applied one dtype to the
    entire mapping)."""
    p = _parent()
    p.register_subframe("FIT", _fit({"p0": [1., 2., 3.]}), index_columns=["run"])
    assert not hasattr(p, "add_aliases")          # the removed method

    formulas = {"c1": "p0 + qpt", "c2": "p0 - qpt"}
    dtypes   = {"c1": "float32", "c2": "float64"}   # per-alias - impossible with add_aliases
    for name, formula in formulas.items():
        p.add_alias(name, formula, source="FIT", dtype=dtypes[name])

    assert p.aliases["c1"] == "FIT.p0 + qpt"
    assert p.aliases["c2"] == "FIT.p0 - qpt"
    p.materialize_aliases(names=["c1", "c2"])
    assert p.df["c1"].dtype == np.dtype("float32")
    assert p.df["c2"].dtype == np.dtype("float64")


def test_T8b_bad_formula_in_a_loop_raises_at_that_formula():
    """Without add_aliases' all-or-nothing guarantee, a bad formula still fails LOUDLY at
    registration (R2), naming the offending token - so a half-bound state is immediately
    visible, not silent."""
    p = _parent()
    p.register_subframe("FIT", _fit({"p0": [1., 2., 3.]}), index_columns=["run"])
    p.add_alias("ok1", "p0 + qpt", source="FIT")
    with pytest.raises(ValueError, match="nosuch"):
        p.add_alias("bad2", "p0 + nosuch", source="FIT")
    assert "ok1" in p.aliases and "bad2" not in p.aliases    # loud, and no bad alias landed


# ---- T-9 [Inv]: eager / lazy / chain parity ----
@pytest.mark.invariance
@pytest.mark.skipif(not _HAS_UPROOT, reason="uproot required for lazy/chain readers")
def test_T9_eager_lazy_chain_parity(tmp_path):
    f0 = str(tmp_path / "a.root"); _parent().export_tree(f0, treename="tree")
    f1 = str(tmp_path / "b.root"); _parent().export_tree(f1, treename="tree")
    coeff = {"p0": [1., 2., 3.], "p1": [10., 20., 30.]}
    outs = []
    for adf in (_parent(),
                AliasDataFrame.read_tree_lazy(f0, tree_name="tree"),
                AliasDataFrame.read_chain_lazy([f0 + ":tree", f1 + ":tree"])):
        adf.register_subframe("FIT", _fit(dict(coeff)), index_columns=["run"])
        adf.add_alias("corr", "p0 + p1*qpt", source="FIT")
        assert adf.aliases["corr"] == "FIT.p0 + FIT.p1*qpt"
        adf.ensure_columns(["qpt", "run"])
        outs.append(adf.eval("corr").values[:N])
    assert np.allclose(outs[0], outs[1])            # eager == lazy
    assert np.allclose(outs[0], outs[2][:N])        # eager == chain (first file)


# ---- T-10: flagship end-to-end - one call == the hand-written aliases (numerically) ----
def test_T10_flagship_matches_handwritten():
    """The one-call source-scoped binding reproduces the hand-written qualified aliases.

    SCOPE (panel GPT18, 2026-07-12): this asserts NUMERICAL equality (np.allclose) on a
    SYNTHETIC representative frame — not bitwise equality, and not apass1 production data.
    Validation against real apass1 data belongs in a usage transcript, not this unit test.
    """
    p = _parent()
    coeffs = {"dcar_intercept": [0.5, 1.5, 2.5], "dcar_slope_qpt": [2., 3., 4.],
              "dcar_slope_tgl": [1., 1., 1.]}
    p.register_subframe("DCABiasFitP2", _fit(coeffs), index_columns=["run"])
    formulas = {"dcar_fit": "dcar_intercept + dcar_slope_qpt*qpt + dcar_slope_tgl*tgl"}
    for _n, _f in formulas.items():                                      # the supported form
        p.add_alias(_n, _f, source="DCABiasFitP2")
    # hand-written equivalent (what users write today)
    p.add_alias("dcar_fit_manual",
                "DCABiasFitP2.dcar_intercept + DCABiasFitP2.dcar_slope_qpt*qpt "
                "+ DCABiasFitP2.dcar_slope_tgl*tgl")
    p.materialize_aliases(names=["dcar_fit", "dcar_fit_manual"])
    assert np.allclose(p.df["dcar_fit"].values, p.df["dcar_fit_manual"].values)  # numerically equal


# ---- T-11 [X-6]: source= composes with a 13.70 vector (group) alias ----
def test_T11_vector_alias_composition():
    p = _parent()
    p.register_subframe("F", _fit({"a": [1., 1., 1.], "b": [2., 2., 2.]}), index_columns=["run"])
    p.register_function("pair", lambda a, b, q: np.column_stack([a * q, b * q]))
    p.add_alias(["u", "w"], "pair(a,b,qpt)", dtype=["float32", "float32"], source="F")
    p.materialize_aliases(names=["u", "w"])
    assert np.allclose(p.df["u"].values, np.arange(N) * 1.0)       # a=1 -> u = qpt
    assert np.allclose(p.df["w"].values, np.arange(N) * 2.0)       # b=2 -> w = 2*qpt


# ---- unknown source subframe is a clear error ----
def test_unknown_source_refused():
    p = _parent()
    with pytest.raises(ValueError, match="NOPE"):
        p.add_alias("x", "p0 + qpt", source="NOPE")
