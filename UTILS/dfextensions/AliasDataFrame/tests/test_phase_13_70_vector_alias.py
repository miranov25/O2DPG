"""PHASE_13_70_ADF — Vector (Group) Aliases — T-VEC suite.

Group aliases evaluate ONE expression once (via the D0 function-generic engine) and
split it into k scalar member columns. These tests pin correctness, evaluate-once,
invalidation, and the V-6 / CF-5 / CF-8 error contract. No ROOT/dfdraw needed.
"""
import numpy as np
import pandas as pd
import pytest

from AliasDataFrame import AliasDataFrame

RTOL, ATOL = 1e-9, 1e-12


@pytest.fixture
def adf():
    rng = np.random.default_rng(0)
    n = 40
    return AliasDataFrame(pd.DataFrame({
        "a": rng.random(n), "b": rng.random(n), "c": rng.random(n),
    }))


def _count_wrap(fn, counter, full_only_n):
    """Wrap a group function to count only FULL-size evaluations (ignores the cheap
    1-row arity probe done at definition)."""
    def _w(*args):
        if len(np.asarray(args[0])) == full_only_n:
            counter["n"] += 1
        return fn(*args)
    return _w


# ------------------------------------------------------------- T-VEC-1 tuple expr
def test_VEC_1_tuple_expression_members(adf):
    adf.add_alias(["s", "d", "t"], "a + b, a - b, a * 2")
    assert all(m in adf.aliases for m in ["s", "d", "t"])
    np.testing.assert_allclose(np.asarray(adf.eval("s")), adf.df["a"] + adf.df["b"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(np.asarray(adf.eval("d")), adf.df["a"] - adf.df["b"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(np.asarray(adf.eval("t")), adf.df["a"] * 2, rtol=RTOL, atol=ATOL)


# ---------------------------------------------- T-VEC-2 2-D function member split
@pytest.mark.invariance
def test_VEC_2_2d_function_members_equal_columns(adf):
    def dist(a, b):
        a, b = np.asarray(a), np.asarray(b)
        return np.column_stack([a + b, a - b])
    adf.register_function("dist", dist, overwrite=True)
    adf.add_alias(["p", "q"], "dist(a, b)")
    ref = dist(adf.df["a"], adf.df["b"])
    np.testing.assert_allclose(np.asarray(adf.eval("p")), ref[:, 0], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(np.asarray(adf.eval("q")), ref[:, 1], rtol=RTOL, atol=ATOL)


# ------------------------------------------------- T-VEC-3 evaluate-once siblings
def test_VEC_3_evaluate_once_across_siblings(adf):
    n = len(adf.df)
    counter = {"n": 0}
    def dist(a, b):
        a, b = np.asarray(a), np.asarray(b)
        return np.column_stack([a + b, a - b])
    adf.register_function("dist", _count_wrap(dist, counter, n), overwrite=True)
    adf.add_alias(["p", "q"], "dist(a, b)")
    counter["n"] = 0                       # isolate from the 1-row arity probe
    _ = adf.eval("p"); _ = adf.eval("q")   # two siblings, same frame
    assert counter["n"] == 1


# ------------------------------------------------- T-VEC-4 invalidation contract
@pytest.mark.invariance
def test_VEC_4_cache_invalidation(adf):
    n = len(adf.df)
    counter = {"n": 0}
    def dist(a, b):
        a, b = np.asarray(a), np.asarray(b)
        return np.column_stack([a + b, a - b])
    adf.register_function("dist", _count_wrap(dist, counter, n), overwrite=True)
    adf.add_alias(["p", "q"], "dist(a, b)")
    counter["n"] = 0
    _ = adf.eval("p")
    adf.dematerialize(drop=["p", "q"]); _ = adf.eval("q")   # benign re-materialize
    assert counter["n"] == 1                                 # evaluate-once survives
    adf["a"] = np.ones(n)                                    # hooked write to an input
    adf.dematerialize(drop=["p", "q"]); v = np.asarray(adf.eval("p"))
    assert counter["n"] == 2                                 # re-ran after invalidation
    np.testing.assert_allclose(v, np.ones(n) + adf.df["b"], rtol=RTOL, atol=ATOL)


# ------------------------------------------------- T-VEC-5 arity mismatch (V-6)
def test_VEC_5_arity_mismatch_refused(adf):
    with pytest.raises(ValueError) as ei:
        adf.add_alias(["x", "y", "z"], "a + b, a - b")   # 3 names, 2 outputs
    assert "mismatch" in str(ei.value).lower() or "V-6" in str(ei.value)


def test_VEC_5_arity_mismatch_2d_func(adf):
    def two(a):
        a = np.asarray(a); return np.column_stack([a, a * 2])   # 2 outputs
    adf.register_function("two", two, overwrite=True)
    with pytest.raises(ValueError):
        adf.add_alias(["x", "y", "z"], "two(a)")               # under 3 names


# ------------------------------------------------- T-VEC-6 collision (CF-5)
def test_VEC_6_collision_each_namespace(adf):
    with pytest.raises(ValueError) as ei:                       # existing column
        adf.add_alias(["a", "new1"], "a, b")
    assert "COLUMN" in str(ei.value) or "collides" in str(ei.value)
    adf.add_alias("solo", "a + 1")                              # existing alias
    with pytest.raises(ValueError) as ei:
        adf.add_alias(["solo", "new2"], "a, b")
    assert "ALIAS" in str(ei.value)
    adf.add_alias(["g1", "g2"], "a, b")                         # another group member
    with pytest.raises(ValueError) as ei:
        adf.add_alias(["g1", "new3"], "a, b")
    assert "GROUP" in str(ei.value)


# ------------------------------------------------- T-VEC-7 dtype-list + duplicates
def test_VEC_7_dtype_length_and_duplicate_names(adf):
    with pytest.raises(ValueError) as ei:
        adf.add_alias(["u", "v"], "a + b, a - b", dtype=["float64"])   # 1 dtype, 2 names
    assert "length" in str(ei.value)
    with pytest.raises(ValueError) as ei:
        adf.add_alias(["w", "w"], "a + b, a - b")                      # duplicate names
    assert "duplicate" in str(ei.value).lower()


def test_VEC_7_dtype_applied(adf):
    adf.add_alias(["i1", "i2"], "a * 100, b * 100", dtype=["int32", "int32"])
    assert np.asarray(adf.eval("i1")).dtype == np.int32


# ------------------------------------------------- T-VEC-8 single-name list
def test_VEC_8_single_name_list_is_ordinary_alias(adf):
    adf.add_alias(["only"], "a + b")
    np.testing.assert_allclose(np.asarray(adf.eval("only")), adf.df["a"] + adf.df["b"], rtol=RTOL, atol=ATOL)
    assert "only" not in getattr(adf, "_group_members", {})   # not a group member


# ------------------------------------------------- T-VEC-9 invariance vs prefilled
@pytest.mark.invariance
def test_VEC_9_member_equals_prefilled_column(adf):
    def dist(a, b):
        a, b = np.asarray(a), np.asarray(b)
        return np.column_stack([a + b, a - b])
    adf.register_function("dist", dist, overwrite=True)
    adf.add_alias(["p", "q"], "dist(a, b)")
    member = np.asarray(adf.eval("p"))
    adf["p_pref"] = dist(adf.df["a"], adf.df["b"])[:, 0]
    np.testing.assert_allclose(member, np.asarray(adf.df["p_pref"]), rtol=RTOL, atol=ATOL)
    e_member = np.asarray(adf.eval("p * 2 + c"))
    e_pref = np.asarray(adf.eval("p_pref * 2 + c"))
    np.testing.assert_allclose(e_member, e_pref, rtol=RTOL, atol=ATOL)


# ------------------------------------------------- T-VEC-10 tuple vs 2-D equivalence
@pytest.mark.invariance
def test_VEC_10_tuple_and_2d_equivalent(adf):
    adf.add_alias(["ta", "tb"], "a + b, a - b")
    def m(a, b):
        a, b = np.asarray(a), np.asarray(b)
        return np.column_stack([a + b, a - b])
    adf.register_function("m", m, overwrite=True)
    adf.add_alias(["na", "nb"], "m(a, b)")
    np.testing.assert_allclose(np.asarray(adf.eval("ta")), np.asarray(adf.eval("na")), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(np.asarray(adf.eval("tb")), np.asarray(adf.eval("nb")), rtol=RTOL, atol=ATOL)


# ------------------------------------------------- T-VEC-11 parquet persistence (D4a)
@pytest.mark.invariance
def test_VEC_11_parquet_roundtrip_recovers_group(tmp_path):
    rng = np.random.default_rng(1)
    n = 12
    adf = AliasDataFrame(pd.DataFrame({"a": rng.random(n), "b": rng.random(n)}))
    adf.add_alias(["s", "dd"], "a + b, a - b", dtype=["float64", "float64"])
    pre_s = np.asarray(adf.eval("s")); pre_d = np.asarray(adf.eval("dd"))
    p = str(tmp_path / "round")
    adf.save(p)
    adf2 = AliasDataFrame.load(p)
    gid = "__grp__s__dd"
    assert gid in adf2._group_registry                     # group recovered
    assert adf2._group_members.get("s") == gid
    np.testing.assert_allclose(np.asarray(adf2.eval("s")), pre_s, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(np.asarray(adf2.eval("dd")), pre_d, rtol=RTOL, atol=ATOL)


# ------------------------------------------------- T-VEC-12 dematerialize all-or-none (D5)
def test_VEC_12_dematerialize_all_or_none(adf):
    adf.add_alias(["s", "dd"], "a + b, a - b")
    _ = adf.eval("s"); _ = adf.eval("dd")
    assert "s" in adf.df.columns and "dd" in adf.df.columns
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        dropped = adf.dematerialize(drop=["s"])          # drop ONE member
    assert set(dropped) == {"s", "dd"}                    # whole group dropped
    assert "dd" not in adf.df.columns
    assert any("all-or-none" in str(x.message) for x in w)


def test_VEC_12b_dematerialize_keep_keeps_group(adf):
    adf.add_alias(["s", "dd"], "a + b, a - b")
    adf.add_alias("solo", "a * 3")
    for m in ["s", "dd", "solo"]:
        _ = adf.eval(m)
    adf.dematerialize(keep=["s"])                          # keep one member -> keep group
    assert "s" in adf.df.columns and "dd" in adf.df.columns
    assert "solo" not in adf.df.columns                   # ordinary alias dropped


# ------------------------------------------------- T-VEC-13 self/sibling cycle (CF-10)
def test_VEC_13_self_cycle_refused(adf):
    with pytest.raises(ValueError) as ei:
        adf.add_alias(["p", "q"], "p + 1, a - b")         # references own member 'p'
    assert "cycle" in str(ei.value).lower() and "CF-10" in str(ei.value)


# ------------------------------------------------- T-VEC-14 V-6 dict / jagged refusals (D6)
def test_VEC_14_dict_return_refused(adf):
    def d(a, b):
        a, b = np.asarray(a), np.asarray(b)
        return {"x": a + b, "y": a - b}                    # dict, not tuple/2-D
    adf.register_function("d", d, overwrite=True)
    with pytest.raises(ValueError) as ei:
        adf.add_alias(["p", "q"], "d(a, b)")
    assert "dict" in str(ei.value) and "V-6" in str(ei.value)


def test_VEC_14b_jagged_return_refused(adf):
    def j(a):
        a = np.asarray(a)
        return (a, a[:-1])                                 # differing lengths
    adf.register_function("j", j, overwrite=True)
    with pytest.raises(ValueError) as ei:
        adf.add_alias(["p", "q"], "j(a)")
    assert "jagged" in str(ei.value) and "V-6" in str(ei.value)


# ------------------------------------------------- T-VEC-4 ROOT persistence (D4b)
_HAS_UPROOT = False
try:
    import uproot as _uproot  # noqa
    _HAS_UPROOT = True
except Exception:
    pass


@pytest.mark.skipif(not _HAS_UPROOT, reason="uproot required for ROOT roundtrip")
@pytest.mark.invariance
def test_VEC_4_root_export_and_read_tree_lazy_recover(tmp_path):
    rng = np.random.default_rng(2)
    n = 16
    adf = AliasDataFrame(pd.DataFrame({"a": rng.random(n), "b": rng.random(n)}))
    adf.add_alias(["s", "dd"], "a + b, a - b", dtype=["float64", "float64"])
    pre_s = np.asarray(adf.eval("s")); pre_d = np.asarray(adf.eval("dd"))
    f = str(tmp_path / "vec.root")
    adf.export_tree(f, treename="tree")
    # write side: registry embedded under ADF_GROUP/
    keys = {k.split(";")[0] for k in _uproot.open(f).keys()}
    assert "ADF_GROUP/registry" in keys
    # read side (uproot lazy reader): group recovers, values equal pre-export
    adf2 = AliasDataFrame.read_tree_lazy(f, tree_name="tree")
    assert "__grp__s__dd" in adf2._group_registry
    adf2.ensure_columns(["a", "b"])
    np.testing.assert_allclose(np.asarray(adf2.eval("s")), pre_s, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(np.asarray(adf2.eval("dd")), pre_d, rtol=RTOL, atol=ATOL)


@pytest.mark.skipif(not _HAS_UPROOT, reason="uproot required")
def test_VEC_4b_direct_recover_from_file(tmp_path):
    rng = np.random.default_rng(3)
    n = 10
    adf = AliasDataFrame(pd.DataFrame({"a": rng.random(n), "b": rng.random(n)}))
    adf.add_alias(["p", "q"], "a * b, a + b")
    pre_p = np.asarray(adf.eval("p"))
    f = str(tmp_path / "vec2.root"); adf.export_tree(f, treename="tree")
    fresh = AliasDataFrame(pd.DataFrame({"a": adf.df["a"].values, "b": adf.df["b"].values}))
    fresh._group_recover_from_file(f)
    assert "__grp__p__q" in fresh._group_registry
    np.testing.assert_allclose(np.asarray(fresh.eval("p")), pre_p, rtol=RTOL, atol=ATOL)


@pytest.mark.skipif(not _HAS_UPROOT, reason="uproot required")
def test_VEC_4c_chain_first_file_canonical_recover(tmp_path):
    rng = np.random.default_rng(4)
    n = 8
    frames = []
    for i in range(2):
        a = rng.random(n); b = rng.random(n)
        adf = AliasDataFrame(pd.DataFrame({"a": a, "b": b}))
        adf.add_alias(["s", "dd"], "a + b, a - b")
        f = str(tmp_path / f"chain_{i}.root"); adf.export_tree(f, treename="tree")
        frames.append(f)
    chained = AliasDataFrame.read_chain_lazy([x + ":tree" for x in frames])
    assert "__grp__s__dd" in getattr(chained, "_group_registry", {})   # first-file-canonical


# ------------------------------------------------- T-VEC-8b release_branches lazy leg (D5)
@pytest.mark.skipif(not _HAS_UPROOT, reason="uproot required for a lazy frame")
def test_VEC_8b_release_branches_lazy_names_group(tmp_path):
    n = 12
    adf = AliasDataFrame(pd.DataFrame({"a": np.arange(n) * 1.0, "b": np.arange(n) + 3.0}))
    adf.add_alias(["s", "dd"], "a + b, a - b")
    f = str(tmp_path / "t.root"); adf.export_tree(f, treename="tree")
    lz = AliasDataFrame.read_tree_lazy(f, tree_name="tree")   # lazy reader present
    assert "__grp__s__dd" in lz._group_registry
    with pytest.raises(ValueError) as ei:
        lz.release_branches(["s"])                            # member is an alias (DD-gamma)
    msg = str(ei.value)
    assert "member of vector/group alias" in msg
    assert "'dd'" in msg and "dematerialize" in msg          # names sibling + directs


# ------------------------------------------------- T-VEC-9 ONNX 2-D flagship
_HAS_ONNX = False
try:
    import onnxruntime as _ort            # noqa
    from sklearn.ensemble import RandomForestRegressor as _RFR   # noqa
    from skl2onnx import convert_sklearn as _cvt                 # noqa
    from skl2onnx.common.data_types import FloatTensorType as _FTT  # noqa
    _HAS_ONNX = True
except Exception:
    pass


@pytest.mark.skipif(not _HAS_ONNX, reason="onnxruntime/sklearn/skl2onnx required")
@pytest.mark.invariance
def test_VEC_9_onnx_2d_flagship(tmp_path):
    rng = np.random.default_rng(0)
    Xt = rng.random((200, 3)).astype(np.float32)
    Yt = np.column_stack([2 * Xt[:, 0] - Xt[:, 1], Xt[:, 1] + 0.5 * Xt[:, 2]]).astype(np.float32)
    model = _RFR(n_estimators=8, max_depth=4).fit(Xt, Yt)
    onx = _cvt(model, initial_types=[("input", _FTT([None, 3]))])
    op = str(tmp_path / "m.onnx"); open(op, "wb").write(onx.SerializeToString())
    sess = _ort.InferenceSession(op)
    in_name = sess.get_inputs()[0].name
    n = 20
    A, B, C = rng.random(n), rng.random(n), rng.random(n)
    adf = AliasDataFrame(pd.DataFrame({"a": A, "b": B, "c": C}))
    calls = {"n": 0}

    def predict_dist(a, b, c):
        calls["n"] += (len(np.asarray(a)) == n)
        X = np.column_stack([np.asarray(a), np.asarray(b), np.asarray(c)]).astype(np.float32)
        return np.asarray(sess.run(None, {in_name: X})[0])    # (n, 2) — the V-6 2-D case

    adf.register_function("predict_dist", predict_dist, overwrite=True)
    adf.add_alias(["dY", "dZ"], "predict_dist(a, b, c)", dtype=["float32", "float32"])
    ref = np.asarray(sess.run(None, {in_name: np.column_stack([A, B, C]).astype(np.float32)})[0])
    calls["n"] = 0
    dy = np.asarray(adf.eval("dY")); dz = np.asarray(adf.eval("dZ"))
    np.testing.assert_allclose(dy, ref[:, 0], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(dz, ref[:, 1], rtol=1e-5, atol=1e-6)
    assert calls["n"] == 1                                    # model runs ONCE for both members
