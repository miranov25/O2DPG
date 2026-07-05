"""PHASE_13_69_ADF — ML Model Store & Inference Interface (T-ML-1..14).

Fixture rule (proposal §5): real files, real trained artifacts — the fixture trains
a tiny xgboost model and exports native JSON AND converted ONNX; never mocks.
Tolerances rtol=1e-6/atol=1e-9 (R6). The whole module skips cleanly where the
runtimes/converter are absent (declared deps installed on alma2 per gate G-2).

Environment-only legs (chain recovery D7, external-reference mode D5b, and the
G-1 draw-surface transcript T-ML-0) are marked/skipped and tracked in the CRR.
"""
import os
import json
import hashlib
import tempfile
import numpy as np
import pandas as pd
import pytest

from AliasDataFrame import AliasDataFrame

# --- dependency gate: skip the module if the ML stack is unavailable -------------
ort = pytest.importorskip("onnxruntime")
xgb = pytest.importorskip("xgboost")
try:
    from onnxmltools.convert import convert_xgboost
    from onnxconverter_common.data_types import FloatTensorType
except Exception:                                   # pragma: no cover
    pytest.skip("onnxmltools/onnxconverter-common not available", allow_module_level=True)

RTOL, ATOL = 1e-6, 1e-9

# dfdraw availability (mirror the suite's guard) — the draw tests skip cleanly where
# dfdraw is not importable (e.g. the coder sandbox); they run on alma2.
try:
    from dfdraw import DFDraw  # noqa: F401
    _HAS_DFDRAW = True
except Exception:
    _HAS_DFDRAW = False


# --------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def artifacts():
    """Train a tiny xgboost regressor; export native JSON and converted ONNX."""
    d = tempfile.mkdtemp()
    rng = np.random.default_rng(0)
    Xtr = rng.random((400, 3)).astype(np.float32)
    ytr = (2 * Xtr[:, 0] - Xtr[:, 1] + 0.5 * Xtr[:, 2]).astype(np.float32)
    model = xgb.XGBRegressor(n_estimators=10, max_depth=3)
    model.fit(Xtr, ytr)
    onnx_path = os.path.join(d, "m.onnx")
    onx = convert_xgboost(model, initial_types=[("input", FloatTensorType([None, 3]))])
    with open(onnx_path, "wb") as f:
        f.write(onx.SerializeToString())
    json_path = os.path.join(d, "m.json")
    model.get_booster().save_model(json_path)
    return {"dir": d, "onnx": onnx_path, "json": json_path, "booster": model.get_booster()}


@pytest.fixture
def data_adf():
    rng = np.random.default_rng(7)
    n = 60
    return AliasDataFrame(pd.DataFrame({
        "a": rng.random(n), "b": rng.random(n), "c": rng.random(n),
        "decoy1": rng.random(n), "decoy2": rng.random(n),
    }))


def _direct_onnx(onnx_path, adf, cols):
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    X = np.column_stack([np.asarray(adf.df[c]) for c in cols]).astype(np.float32)
    return sess.run(None, {sess.get_inputs()[0].name: X})[0].ravel()


# ------------------------------------------------------------- T-ML-1 / T-ML-2 (ONNX)
def test_ML_1_register_creates_usable_lazy_alias(artifacts, data_adf):
    data_adf.register_model("pred", artifacts["onnx"], inputs=["a", "b", "c"])
    assert "pred" in data_adf.aliases                 # alias created in one call
    out = np.asarray(data_adf.eval("pred"))
    assert out.shape[0] == len(data_adf.df)


def test_ML_2_prediction_equals_direct_onnxruntime(artifacts, data_adf):
    data_adf.register_model("pred", artifacts["onnx"], inputs=["a", "b", "c"])
    got = np.asarray(data_adf.eval("pred"))
    ref = _direct_onnx(artifacts["onnx"], data_adf, ["a", "b", "c"])
    np.testing.assert_allclose(got, ref, rtol=RTOL, atol=ATOL)


def test_ML_2_native_json_path_matches_booster(artifacts, data_adf):
    data_adf.register_model("predJ", artifacts["json"], inputs=["a", "b", "c"])
    got = np.asarray(data_adf.eval("predJ"))
    X = np.column_stack([data_adf.df[c] for c in ["a", "b", "c"]]).astype(np.float32)
    ref = artifacts["booster"].predict(xgb.DMatrix(X))
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-6)


# --------------------------------------------------------- T-ML-3 surface symmetry
def test_ML_3_surface_symmetry_with_ordinary_alias(artifacts, data_adf):
    """The ML alias behaves like any function-backed alias on the ADF surface: it
    composes in a compound expression and is listed like any alias. (Surface
    symmetry, NOT numerical equality with GB.)"""
    data_adf.register_model("pred", artifacts["onnx"], inputs=["a", "b", "c"])
    base = np.asarray(data_adf.eval("pred"))
    combined = np.asarray(data_adf.eval("pred + b"))    # composes like any alias
    np.testing.assert_allclose(combined, base + np.asarray(data_adf.df["b"]),
                               rtol=RTOL, atol=ATOL)
    assert "pred" in data_adf.aliases                   # listed like any alias


# ----------------------------------------------------- T-ML-4 embed roundtrip (D5a)
@pytest.mark.parametrize("reader", ["eager", "lazy"])
def test_ML_4_embed_roundtrip(artifacts, data_adf, reader):
    if reader == "eager":
        # Eager read_tree reads metadata via PyROOT. Check PyROOT availability
        # DIRECTLY (import ROOT) — layout-independent. Importing ROOT from the
        # AliasDataFrame namespace is fragile: under the package layout the name
        # lives in the module, not the package __init__, so `from AliasDataFrame
        # import ROOT` raises ImportError and masquerades as a flaky failure.
        try:
            import ROOT as _ROOT  # noqa: F401
        except Exception:
            pytest.skip("eager read_tree metadata read needs PyROOT (absent here)")
    data_adf.register_model("pred", artifacts["onnx"], inputs=["a", "b", "c"])
    pre = np.asarray(data_adf.eval("pred")).copy()
    root = os.path.join(artifacts["dir"], f"embed_{reader}.root")
    data_adf.export_tree(root, treename="tree")
    if reader == "eager":
        adf2 = AliasDataFrame.read_tree(root, "tree")
    else:
        adf2 = AliasDataFrame.read_tree_lazy(root, "tree")
    assert "pred" in adf2._models and "pred" in adf2.aliases   # recovered + re-aliased
    post = np.asarray(adf2.eval("pred"))
    np.testing.assert_allclose(post, pre, rtol=RTOL, atol=ATOL)


# ------------------------------------------------- T-ML-6 load-from-ROOT (D1/R5)
def test_ML_6_load_from_root_and_missing_name(artifacts, data_adf):
    data_adf.register_model("pred", artifacts["onnx"], inputs=["a", "b", "c"])
    root = os.path.join(artifacts["dir"], "container.root")
    data_adf.export_tree(root, treename="tree")
    # fresh ADF, register_model(file=root, model=...) pulls the embedded model
    fresh = AliasDataFrame(data_adf.df.copy())
    fresh.register_model("pred", root, model="pred", inputs=["a", "b", "c"])
    np.testing.assert_allclose(
        np.asarray(fresh.eval("pred")),
        _direct_onnx(artifacts["onnx"], fresh, ["a", "b", "c"]), rtol=RTOL, atol=ATOL)
    with pytest.raises(ValueError) as ei:
        fresh.register_model("nope", root, model="not_there", inputs=["a", "b", "c"],
                             overwrite=True)
    assert "not_there" in str(ei.value)


# ------------------------------------------------------------- T-ML-8 error contract
def test_ML_8_duplicate_refuse(artifacts, data_adf):
    data_adf.register_model("pred", artifacts["onnx"], inputs=["a", "b", "c"])
    with pytest.raises(ValueError) as ei:
        data_adf.register_model("pred", artifacts["onnx"], inputs=["a", "b", "c"])
    assert "overwrite" in str(ei.value)
    data_adf.register_model("pred", artifacts["onnx"], inputs=["a", "b", "c"], overwrite=True)


def test_ML_8_unresolvable_auto_format(artifacts, data_adf):
    junk = os.path.join(artifacts["dir"], "junk.bin")
    with open(junk, "wb") as f:
        f.write(b"\x00\x01\x02not a model\x99")
    with pytest.raises(Exception):
        data_adf.register_model("j", junk, inputs=["a", "b", "c"])


def test_ML_8_subframe_input_deferral(artifacts, data_adf):
    sub = AliasDataFrame(pd.DataFrame({"a": [0, 1], "v": [9, 9]}))
    data_adf.register_subframe("S", sub, index_columns="a")
    with pytest.raises(ValueError) as ei:
        data_adf.register_model("p", artifacts["onnx"], inputs=["a", "S.v", "c"])
    assert "DEFERRED" in str(ei.value) or "Subframe" in str(ei.value)


def test_ML_8_md5_mismatch_refuse(artifacts, data_adf):
    """Corrupt an embedded descriptor's MD5 → recovery refuses naming the mismatch."""
    root = os.path.join(artifacts["dir"], "bad.root")
    AliasDataFrame(data_adf.df.copy()).export_tree(root, "tree")
    import uproot
    with uproot.update(root) as fo:
        fo["ADF_ML/x__descriptor"] = json.dumps({
            "name": "x", "format": "onnx", "md5": "deadbeef",
            "inputs": ["a", "b", "c"], "outputs": None, "location": "EMBEDDED"})
        fo["ADF_ML/x__blob"] = {"b": np.frombuffer(open(artifacts["onnx"], "rb").read(), np.uint8)}
    with pytest.raises(ValueError) as ei:
        AliasDataFrame.read_tree_lazy(root, "tree")
    assert "MD5 mismatch" in str(ei.value)


def test_ML_8_missing_runtime_message_shape(artifacts, data_adf, monkeypatch):
    """The missing-runtime path raises RuntimeError with an install hint. We can't
    uninstall onnxruntime here, so assert the handle builder's refuse directly."""
    import builtins
    real_import = builtins.__import__
    def blocked(name, *a, **k):
        if name == "onnxruntime":
            raise ImportError("blocked")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(RuntimeError) as ei:
        AliasDataFrame._ml_make_handle(open(artifacts["onnx"], "rb").read(), "onnx")
    assert "onnxruntime" in str(ei.value) and "install" in str(ei.value).lower()


# --------------------------------------------------------- T-ML-9 lazy exact-load
def test_ML_9_lazy_exact_load_of_input_closure(artifacts, tmp_path):
    import uproot
    rng = np.random.default_rng(5)
    n = 50
    root = str(tmp_path / "feat.root")
    with uproot.recreate(root) as fo:
        fo["tree"] = {k: rng.random(n) for k in ["a", "b", "c", "decoy1", "decoy2", "decoy3"]}
    adf = AliasDataFrame.read_tree_lazy(root, "tree")
    adf.register_model("pred", artifacts["onnx"], inputs=["a", "b", "c"])
    _ = adf.eval("pred")
    loaded = set(adf.loaded_branches)
    assert {"a", "b", "c"} <= loaded                    # inputs loaded
    assert not ({"decoy1", "decoy2", "decoy3"} & loaded)  # decoys untouched


# ------------------------------------------------ T-ML-10 multi-output predict-once
def test_ML_10_multi_output_predict_once(artifacts, data_adf):
    data_adf.register_model("mo", artifacts["onnx"], inputs=["a", "b", "c"],
                            outputs={"variable": "outA"})
    calls = {"n": 0}
    handle = data_adf._models["mo"]["_handle"]
    orig = handle.run
    handle.run = lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1) or orig(*a, **k))
    _ = np.asarray(data_adf.eval("outA"))
    _ = np.asarray(data_adf.eval("outA"))               # cache hit, no second run
    assert calls["n"] == 1


# ------------------------------------------------ T-ML-11 cache invalidation
def test_ML_11_cache_invalidation_on_write(artifacts, data_adf):
    """Predict-once survives benign re-materialize; a hooked write to an input
    invalidates the prediction cache so the next evaluation re-runs the model."""
    data_adf.register_model("pred", artifacts["onnx"], inputs=["a", "b", "c"])
    calls = {"n": 0}
    handle = data_adf._models["pred"]["_handle"]
    orig = handle.run
    handle.run = lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1) or orig(*a, **k))
    v0 = np.asarray(data_adf.eval("pred")).copy()
    data_adf.dematerialize(drop=["pred"]); _ = data_adf.eval("pred")
    assert calls["n"] == 1                               # predict-once across re-materialize
    data_adf["a"] = np.ones(len(data_adf.df))            # hooked write to an input
    data_adf.dematerialize(drop=["pred"]); v1 = np.asarray(data_adf.eval("pred"))
    assert calls["n"] == 2                               # re-ran after invalidation
    ref = _direct_onnx(artifacts["onnx"], data_adf, ["a", "b", "c"])
    np.testing.assert_allclose(v1, ref, rtol=RTOL, atol=ATOL)
    assert not np.allclose(v0, v1)


def test_ML_11_cache_invalidation_on_release(artifacts, tmp_path):
    import uproot
    rng = np.random.default_rng(9); n = 40
    root = str(tmp_path / "r.root")
    with uproot.recreate(root) as fo:
        fo["tree"] = {k: rng.random(n) for k in ["a", "b", "c"]}
    adf = AliasDataFrame.read_tree_lazy(root, "tree")
    adf.register_model("pred", artifacts["onnx"], inputs=["a", "b", "c"])
    _ = adf.eval("pred")
    assert "pred" in adf._model_cache
    adf.dematerialize(drop=["pred"])
    adf.release_branches(["a"])                          # release an input
    assert "pred" not in adf._model_cache               # cache invalidated


# ------------------------------------------------------------- T-ML-12 deregister
def test_ML_12_deregister_full_cleanup_and_reregister(artifacts, data_adf):
    data_adf.register_model("pred", artifacts["onnx"], inputs=["a", "b", "c"])
    _ = data_adf.eval("pred")
    data_adf.deregister_model("pred")
    assert "pred" not in data_adf._models
    assert "pred" not in data_adf.aliases
    assert "pred" not in data_adf._model_cache
    assert "__ml_pred" not in getattr(data_adf, "_registered_functions", {})
    # clean re-registration then succeeds
    data_adf.register_model("pred", artifacts["onnx"], inputs=["a", "b", "c"])
    assert "pred" in data_adf.aliases
    with pytest.raises(ValueError):
        data_adf.deregister_model("never_registered")


# --------------------------------------------------------------- T-ML-13 save_model
def test_ML_13_save_model_canonical_bytes(artifacts, data_adf):
    data_adf.register_model("pred", artifacts["onnx"], inputs=["a", "b", "c"])
    out = os.path.join(artifacts["dir"], "saved.onnx")
    data_adf.save_model("pred", out)
    written_md5 = hashlib.md5(open(out, "rb").read()).hexdigest()
    assert written_md5 == data_adf._models["pred"]["md5"]
    # fresh register on the saved file => identical predictions
    data_adf.register_model("pred2", out, inputs=["a", "b", "c"])
    np.testing.assert_allclose(
        np.asarray(data_adf.eval("pred2")), np.asarray(data_adf.eval("pred")),
        rtol=RTOL, atol=ATOL)


# ----------------------------------------------------------- T-ML-14 format auto
def test_ML_14_format_auto_disambiguation(artifacts, data_adf):
    # ONNX (protobuf) vs native xgboost-JSON, both via auto
    data_adf.register_model("o", artifacts["onnx"], inputs=["a", "b", "c"])
    data_adf.register_model("j", artifacts["json"], inputs=["a", "b", "c"])
    assert data_adf._models["o"]["format"] == "onnx"
    assert data_adf._models["j"]["format"] == "xgboost-json"
    # ROOT container recognized as such (embedded model)
    root = os.path.join(artifacts["dir"], "auto.root")
    data_adf.export_tree(root, "tree")
    with pytest.raises(ValueError) as ei:                # multi/anon container needs model=
        data_adf.register_model("fromroot", root, inputs=["a", "b", "c"], overwrite=True)
    assert "model=" in str(ei.value)
    # explicit format= override honored
    data_adf.register_model("o2", artifacts["onnx"], format="onnx",
                            inputs=["a", "b", "c"], overwrite=True)
    assert data_adf._models["o2"]["format"] == "onnx"


# ------------------------------------- T-ML-15 framework generality (scikit-learn)
def test_ML_15_scikit_randomforest_via_onnx(data_adf, tmp_path):
    """ONNX is the canonical format, so ANY framework that exports ONNX rides the
    same path with no ADF code change. Demonstrated with a scikit-learn
    RandomForestRegressor. (sklearn's NATIVE format is pickle and is deliberately
    NOT supported — arbitrary-code-execution risk; sklearn is supported via ONNX.)"""
    pytest.importorskip("sklearn")
    pytest.importorskip("skl2onnx")
    from sklearn.ensemble import RandomForestRegressor
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    rng = np.random.default_rng(11)
    Xtr = rng.random((300, 3)).astype(np.float32)
    ytr = (2 * Xtr[:, 0] - Xtr[:, 1] + 0.5 * Xtr[:, 2]).astype(np.float32)
    rf = RandomForestRegressor(n_estimators=12, max_depth=4, random_state=0).fit(Xtr, ytr)
    onx = convert_sklearn(rf, initial_types=[("input", FloatTensorType([None, 3]))])
    path = str(tmp_path / "rf.onnx")
    with open(path, "wb") as f:
        f.write(onx.SerializeToString())
    data_adf.register_model("rf", path, inputs=["a", "b", "c"])   # zero code change
    assert data_adf._models["rf"]["format"] == "onnx"             # auto-sniffed
    got = np.asarray(data_adf.eval("rf"))
    ref = _direct_onnx(path, data_adf, ["a", "b", "c"])
    np.testing.assert_allclose(got, ref, rtol=RTOL, atol=ATOL)


# ------------------------------------- T-ML-16 invariance: alias vs prefilled column
def test_ML_16_invariance_alias_vs_prefilled_column(artifacts, data_adf):
    """The ML-backed alias column must be INDISTINGUISHABLE from a plain
    materialized column carrying the same values — numerically and in any
    downstream expression. This pins that the prediction alias is an ordinary
    column-valued alias, not a special-cased object."""
    data_adf.register_model("pred", artifacts["onnx"], inputs=["a", "b", "c"])
    alias_vals = np.asarray(data_adf.eval("pred"))
    direct = _direct_onnx(artifacts["onnx"], data_adf, ["a", "b", "c"])
    data_adf["pred_prefilled"] = direct                          # plain column, same values
    np.testing.assert_allclose(alias_vals, np.asarray(data_adf.df["pred_prefilled"]),
                               rtol=RTOL, atol=ATOL)
    # downstream-expression invariance: expr over the alias == expr over the plain column
    e_alias = np.asarray(data_adf.eval("pred * 2 + b"))
    e_pref = np.asarray(data_adf.eval("pred_prefilled * 2 + b"))
    np.testing.assert_allclose(e_alias, e_pref, rtol=RTOL, atol=ATOL)


# --------------------------------- T-ML-17 per-row vector output refuses loudly
def test_ML_17_per_row_vector_output_refused(data_adf, tmp_path):
    """A single output tensor of width K>1 (a per-row vector) is scalar-incompatible
    with ADF aliases and is deferred in PHASE_13_69 — it must REFUSE loudly at
    registration (the declared shape is unreliable, so the width is probed), not
    fail with a confusing downstream error."""
    pytest.importorskip("sklearn")
    pytest.importorskip("skl2onnx")
    from sklearn.ensemble import RandomForestRegressor
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    rng = np.random.default_rng(13)
    X = rng.random((200, 3)).astype(np.float32)
    Y = np.column_stack([2 * X[:, 0] - X[:, 1], X[:, 2] + X[:, 0]]).astype(np.float32)
    rf = RandomForestRegressor(n_estimators=8, max_depth=4, random_state=0).fit(X, Y)
    onx = convert_sklearn(rf, initial_types=[("input", FloatTensorType([None, 3]))])
    path = str(tmp_path / "multi.onnx")
    with open(path, "wb") as f:
        f.write(onx.SerializeToString())
    with pytest.raises(ValueError) as ei:
        data_adf.register_model("mo", path, inputs=["a", "b", "c"])
    msg = str(ei.value)
    assert "vector" in msg and ("width 2" in msg or "width" in msg)
    assert "mo" not in data_adf.aliases                 # no half-registered alias left


# --------------------------------- T-ML-18 / T-ML-19 the drawing tests (G-1 essence)
# The point: a prediction alias is an ordinary function-backed alias, so it draws
# like any variable and works in the draw slots with NO new draw/parser code.
def _draw_adf(artifacts):
    rng = np.random.default_rng(21)
    n = 120
    adf = AliasDataFrame(pd.DataFrame({
        "a": rng.random(n), "b": rng.random(n), "c": rng.random(n),
        "grp": np.tile([0, 1, 2], n // 3),          # a discrete column to facet on
    }))
    adf.register_model("pred", artifacts["onnx"], inputs=["a", "b", "c"])
    # NB: the prediction is NOT pre-materialized. The draw calls below pass
    # lazy=True — draw() then materializes the function-backed ML alias on demand
    # via its lazy path (the representative workflow). This exercises the G-1
    # premise directly: the ML alias flows through draw with no new dispatch code.
    return adf


@pytest.mark.skipif(not _HAS_DFDRAW, reason="Requires dfdraw")
def test_ML_18_draw_prediction_alias(artifacts):
    """Trivial: the ML prediction draws like any variable — a histogram of `pred`
    and a profile of `pred` versus an input. Drawn with the lazy switch, so draw()
    materializes the prediction alias on demand. Must produce a figure, not raise."""
    adf = _draw_adf(artifacts)
    res_hist = adf.draw("pred", type="hist", bins=20, lazy=True)
    assert res_hist is not None
    fig, ax, stats = adf.draw("pred:a", type="profile", bins=5, min_entries=1, lazy=True)
    assert ax is not None


@pytest.mark.skipif(not _HAS_DFDRAW, reason="Requires dfdraw")
def test_ML_19_ml_alias_in_draw_slots(artifacts):
    """The ML alias works in the draw slots — weights= and facet_by= — exactly like
    a GB-evaluator alias (the G-1 premise: function-backed aliases inherit all draw
    slots)."""
    adf = _draw_adf(artifacts)
    res_w = adf.draw("a", type="hist", weights="pred", bins=10, lazy=True)   # ML alias as weights
    assert res_w is not None
    res_f = adf.draw("pred:a", type="profile", facet_by="grp",    # faceted, prediction drawn
                     bins=5, min_entries=1, lazy=True)
    assert res_f is not None
