"""
PHASE_13_79_ADF B-2 / B-4
==========================

B-2: systematic invariance grid paired to the ratified B-0 / B-1 surface.
B-4: bounded vector × facet/group interaction seams.

The reference bypasses AliasDataFrame preparation entirely: it runs the same
logical request on a plain pandas DataFrame through dfdraw directly.  This
makes B-2 an ADF integration oracle rather than a second copy of ADF's own
alias/subframe/lazy preparation path.
"""
from __future__ import annotations

import tempfile
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dfdraw.drawer import DFDraw
from AliasDataFrame import AliasDataFrame
from tests import test_phase_13_79_slot_grid as B1

uproot = pytest.importorskip("uproot")

CELLS = B1.CELLS
PASSING_CELLS = tuple(row for row in CELLS if row["current_state"] == "PASSING")
KNOWN_GAP_CELLS = tuple(row for row in CELLS if row["current_state"] == "KNOWN_GAP")
INVARIANCE_CASES = CELLS


def _id(row):
    return B1._cell_pytest_id(row)


def _arrays(n=120):
    i = np.arange(n, dtype=np.int32)
    x = np.linspace(-2.4, 2.4, n, dtype=np.float64)
    aux = ((i % 9) - 4).astype(np.float64) / 13.0
    return {
        "k": i,
        "x": x,
        "value": 1.7 + 0.63 * x + 0.17 * aux,
        "aux": aux,
        "sel": (i % 4 != 0),
        "sel0": (i % 3 != 0),
        "sel1": (i % 5 <= 2),
        "weight": 0.8 + (i % 7).astype(np.float64) * 0.09,
        "w0": 0.7 + (i % 5).astype(np.float64) * 0.13,
        "w1": 1.1 + (i % 7).astype(np.float64) * 0.07,
        "group": (i % 3).astype(np.int16),
        "facet": (i % 4).astype(np.int16),
        "color": 0.25 + (i % 11).astype(np.float64),
    }


STRUCT_MEMBERS = list(B1.STRUCT_MEMBERS)


def _eager_main_dataframe():
    data = _arrays()
    df = pd.DataFrame(data)
    for member in STRUCT_MEMBERS:
        source = member.removeprefix("s_")
        df[f"{member}__st"] = np.asarray(data[source])
    return df


def _plain_dataframe():
    return pd.DataFrame(_arrays())


def _child_dataframe():
    data = _arrays()
    members = sorted({name for names in B1.TARGETS.values() for name in names})
    return pd.DataFrame({"k": data["k"], **{m: np.asarray(data[m]) for m in members}})


class _TrackingLazyReader:
    def __init__(self, data):
        self.data = data.copy()
        self.available_branches = set(data.columns)
        self.loaded_branches = set()
        self.num_entries = len(data)
        self.adf_metadata = None

    def load_branches(self, names):
        names = set(names)
        missing = names - self.available_branches
        if missing:
            raise ValueError(f"missing B2 tracking branches: {sorted(missing)}")
        to_load = names - self.loaded_branches
        self.loaded_branches.update(to_load)
        if not to_load:
            return pd.DataFrame(index=self.data.index)
        return self.data[sorted(to_load)].copy()


def _lazy_raw(*, include_struct=False):
    data = _arrays()
    raw = pd.DataFrame(data)
    if include_struct:
        for member in STRUCT_MEMBERS:
            source = member.removeprefix("s_")
            raw[f"st/{member}"] = np.asarray(data[source])
    return raw


@pytest.fixture(scope="session")
def _root_files(tmp_path_factory):
    root = tmp_path_factory.mktemp("phase_13_79_b2")
    data = _arrays()
    child_path = root / "child.root"
    members = sorted({name for names in B1.TARGETS.values() for name in names})
    tree = {"k": data["k"], **{m: np.asarray(data[m]) for m in members}}
    with uproot.recreate(child_path) as f:
        f.mktree("tree", {name: arr.dtype for name, arr in tree.items()})
        f["tree"].extend(tree)
    return None, child_path


def _make_adf(row, root_files):
    _, child_path = root_files
    mode = row["mode"]
    form = row["form"]

    if mode == "eager":
        adf = AliasDataFrame(_eager_main_dataframe())
        adf.draw_lazy = False
    elif mode == "lazy":
        raw = _lazy_raw(include_struct=(form == "struct"))
        adf = AliasDataFrame(pd.DataFrame(index=range(len(raw))))
        adf._lazy_reader = _TrackingLazyReader(raw)
        adf._chain = {
            "files": [], "entry_offsets": [0], "total_entries": len(raw),
            "validation_mode": None,
        }
        adf.draw_lazy = True
    elif mode == "eager_parent_lazy_child":
        adf = AliasDataFrame(_eager_main_dataframe())
        adf.draw_lazy = True
    else:
        raise AssertionError(mode)

    if form == "alias":
        for name, expr in B1.ALIAS_EXPR.items():
            adf.add_alias(name, expr)

    if form == "subframe":
        if mode == "eager_parent_lazy_child":
            adf.register_subframe_lazy(
                "S", str(child_path), tree_name="tree",
                index_columns=["k"], alignment="1:1",
            )
        else:
            adf.register_subframe("S", AliasDataFrame(_child_dataframe()), index_columns=["k"])
            if mode == "lazy":
                adf.ensure_branches(["k"])

    if form == "struct":
        if "st" not in getattr(adf, "_structs", {}):
            adf.register_struct("st", STRUCT_MEMBERS)

    return adf


def _logical_target(slot, form):
    names = B1.TARGETS[slot]
    if form == "column":
        vals = list(names)
    elif form == "alias":
        vals = [f"{name}_alias" for name in names]
    elif form == "expression":
        mapping = {
            "value": "value+0*aux",
            "sel": "(sel)&(aux>-99)",
            "sel0": "(sel0)&(aux>-99)",
            "sel1": "(sel1)&(aux>-99)",
            "weight": "weight*(1.0+0*aux)",
            "w0": "w0*(1.0+0*aux)",
            "w1": "w1*(1.0+0*aux)",
            "group": "group+0",
            "facet": "facet+0",
            "color": "color+0*aux",
        }
        vals = [mapping[n] for n in names]
    elif form == "subframe":
        vals = [f"S.{n}" for n in names]
    elif form == "struct":
        vals = [f"st.s_{n}" for n in names]
    else:
        raise AssertionError(form)
    return vals[0] if len(vals) == 1 else vals


def _plain_target(slot):
    names = B1.TARGETS[slot]
    return names[0] if len(names) == 1 else list(names)


def _materialize_eager_alias(adf, slot):
    names = B1.TARGETS[slot]
    adf.materialize_aliases(names=[f"{n}_alias" for n in names])


def _request(slot, target):
    kwargs = {"auto_title": False}
    if slot == "expr":
        return target, {**kwargs, "type": "hist", "bins": 12}
    if slot == "color":
        return "value:x", {**kwargs, "type": "scatter", "color": target}
    kwargs.update({"type": "profile", "bins": 12, "return_data": True, slot: target})
    if slot in B1.VECTOR_SLOTS:
        kwargs["vector_compose"] = "outer"
    return "value:x", kwargs


def _candidate(row, root_files):
    adf = _make_adf(row, root_files)
    if row["form"] == "alias" and row["mode"] == "eager":
        _materialize_eager_alias(adf, row["slot"])
    expr, kwargs = _request(row["slot"], _logical_target(row["slot"], row["form"]))
    return adf.draw(expr, **kwargs)


def _reference(row):
    expr, kwargs = _request(row["slot"], _plain_target(row["slot"]))
    return DFDraw(_plain_dataframe()).draw(expr, **kwargs)


def _numeric_array(x):
    return np.asarray(x, dtype=np.float64)


def _assert_array_equalish(a, b, *, label, atol=1e-12, rtol=1e-12):
    aa = np.asarray(a)
    bb = np.asarray(b)
    assert aa.shape == bb.shape, f"{label}: shape {aa.shape} != {bb.shape}"
    if np.issubdtype(aa.dtype, np.number) and np.issubdtype(bb.dtype, np.number):
        np.testing.assert_allclose(aa.astype(float), bb.astype(float), rtol=rtol, atol=atol, equal_nan=True,
                                   err_msg=label)
    else:
        assert aa.tolist() == bb.tolist(), f"{label}: {aa.tolist()} != {bb.tolist()}"


def _assert_frame_semantic_equal(a, b, *, label):
    assert isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame), label
    assert list(a.columns) == list(b.columns), f"{label}: columns differ"
    assert a.shape == b.shape, f"{label}: shape {a.shape} != {b.shape}"
    for col in a.columns:
        _assert_array_equalish(a[col].to_numpy(), b[col].to_numpy(), label=f"{label}.{col}")


def _profile_evidence(stats):
    assert isinstance(stats, dict)
    frame = stats.get("profile_data")
    assert isinstance(frame, pd.DataFrame) and not frame.empty
    return frame.reset_index(drop=True)


def _facet_evidence(stats):
    assert isinstance(stats, dict) and stats.get("faceted") is True
    groups = list(stats["groups"])
    per = stats["per_group"]
    frames = []
    for g in groups:
        sub = per[str(g)]
        frames.append((g, _profile_evidence(sub)))
    return groups, frames


def _hist_evidence(result):
    fig, ax, stats = result
    geometry = []
    for patch in ax.patches:
        if hasattr(patch, "get_xy"):
            geometry.append(np.asarray(patch.get_xy(), dtype=float))
        else:
            geometry.append(np.asarray([
                [float(patch.get_x()), float(patch.get_height())],
                [float(patch.get_x()) + float(patch.get_width()), float(patch.get_height())],
            ]))
    return {
        "n": stats["n"], "mean": stats["mean"], "std": stats["std"],
        "min": stats["min"], "max": stats["max"], "geometry": geometry,
    }


def _color_evidence(result):
    fig, ax, stats = result
    assert ax.collections, "color oracle: scatter has no collection"
    coll = ax.collections[0]
    arr = coll.get_array()
    assert arr is not None
    offsets = coll.get_offsets()
    return {
        "n": stats["n"],
        "colors": np.asarray(arr),
        "offsets": np.asarray(offsets),
    }


def _assert_non_degenerate_reference(row, result):
    slot = row["slot"]
    stats = result[2]
    if slot == "expr":
        ev = _hist_evidence(result)
        assert ev["geometry"], "hist oracle: no rendered histogram geometry"
        y = np.concatenate([g[:, 1] for g in ev["geometry"]])
        assert np.count_nonzero(y) >= 3 and np.nanmax(y) > 0
        assert ev["max"] > ev["min"]
    elif slot == "selection":
        assert 0 < stats["n"] < len(_arrays()["x"])
        assert len(_profile_evidence(stats)) >= 4
    elif slot == "selection_vector":
        assert isinstance(stats, list) and len(stats) == 2
        assert all(0 < s["n"] < len(_arrays()["x"]) for s in stats)
        assert stats[0]["n"] != stats[1]["n"] or not np.allclose(
            _profile_evidence(stats[0])["y_mean"], _profile_evidence(stats[1])["y_mean"], equal_nan=True)
    elif slot == "weights":
        pf = _profile_evidence(stats)
        assert "sum_weights" in pf.columns
        assert np.ptp(_arrays()["weight"]) > 0
    elif slot == "weights_vector":
        assert isinstance(stats, list) and len(stats) == 2
        assert all("sum_weights" in _profile_evidence(s).columns for s in stats)
        assert not np.allclose(
            _profile_evidence(stats[0])["sum_weights"], _profile_evidence(stats[1])["sum_weights"], equal_nan=True)
    elif slot == "group_by":
        pf = _profile_evidence(stats)
        assert set(pf["group"].dropna().tolist()) == {0, 1, 2}
    elif slot == "facet_by":
        groups, frames = _facet_evidence(stats)
        assert groups == [0, 1, 2, 3]
        assert all(len(frame) >= 4 for _, frame in frames)
    elif slot == "color":
        ev = _color_evidence(result)
        assert len(np.unique(ev["colors"])) >= 5
        assert ev["offsets"].shape[0] == len(_arrays()["x"])
    else:
        raise AssertionError(slot)


def _assert_semantic_equal(row, cand, ref):
    slot = row["slot"]
    if slot == "expr":
        a, b = _hist_evidence(cand), _hist_evidence(ref)
        for key in ("n", "mean", "std", "min", "max"):
            _assert_array_equalish([a[key]], [b[key]], label=f"{row['cell_id']}.{key}")
        assert len(a["geometry"]) == len(b["geometry"]), f"{row['cell_id']}.hist_geometry count"
        for i, (ga, gb) in enumerate(zip(a["geometry"], b["geometry"])):
            _assert_array_equalish(ga, gb, label=f"{row['cell_id']}.hist_geometry[{i}]")
        return

    if slot == "color":
        a, b = _color_evidence(cand), _color_evidence(ref)
        assert a["n"] == b["n"]
        _assert_array_equalish(a["colors"], b["colors"], label=f"{row['cell_id']}.color_values")
        _assert_array_equalish(a["offsets"], b["offsets"], label=f"{row['cell_id']}.scatter_offsets")
        return

    sa, sb = cand[2], ref[2]
    if slot in {"selection_vector", "weights_vector"}:
        assert isinstance(sa, list) and isinstance(sb, list)
        assert len(sa) == len(sb) == 2, f"{row['cell_id']}: vector branch count"
        for i, (xa, xb) in enumerate(zip(sa, sb)):
            assert xa["n"] == xb["n"], f"{row['cell_id']}.branch{i}.n"
            _assert_frame_semantic_equal(_profile_evidence(xa), _profile_evidence(xb),
                                         label=f"{row['cell_id']}.branch{i}.profile")
        return

    if slot == "facet_by":
        ga, fa = _facet_evidence(sa)
        gb, fb = _facet_evidence(sb)
        assert ga == gb, f"{row['cell_id']}: facet identity/order"
        for (g1, a), (g2, b) in zip(fa, fb):
            assert g1 == g2
            _assert_frame_semantic_equal(a, b, label=f"{row['cell_id']}.facet[{g1}]")
        return

    assert sa["n"] == sb["n"], f"{row['cell_id']}.n"
    _assert_frame_semantic_equal(_profile_evidence(sa), _profile_evidence(sb),
                                 label=f"{row['cell_id']}.profile")


def _param(row):
    if row["current_state"] != "KNOWN_GAP":
        return pytest.param(row, id=_id(row))
    return pytest.param(
        row, id=_id(row),
        marks=pytest.mark.xfail(strict=True, reason=f"{row['owning_bug']} | B-2 blocked by live B-1 gap"),
    )


PARAMS = tuple(_param(r) for r in INVARIANCE_CASES)


@pytest.mark.parametrize("row", PARAMS)
def test_slot_grid_invariance(row, _root_files):
    if row["current_state"] == "KNOWN_GAP":
        try:
            _candidate(row, _root_files)
        except Exception as exc:
            B1._assert_gap_signature(row, exc)
            raise
        return

    ref = _reference(row)
    cand = None
    try:
        _assert_non_degenerate_reference(row, ref)
        cand = _candidate(row, _root_files)
        _assert_semantic_equal(row, cand, ref)
    finally:
        plt.close("all")


PASSING_BASE_PAIRS = tuple(
    (slot, form)
    for slot, form in product(B1.SLOTS, B1.FORMS)
    if all(B1.CELL_BY_ID[f"slotgrid:{slot}:{form}:{mode}"]["current_state"] == "PASSING"
           for mode in B1.BASE_MODES)
)


@pytest.mark.parametrize("slot,form", PASSING_BASE_PAIRS,
                         ids=lambda x: x if isinstance(x, str) else str(x))
def test_slot_grid_cross_mode(slot, form, _root_files):
    eager = B1.CELL_BY_ID[f"slotgrid:{slot}:{form}:eager"]
    lazy = B1.CELL_BY_ID[f"slotgrid:{slot}:{form}:lazy"]
    re = _candidate(eager, _root_files)
    rl = _candidate(lazy, _root_files)
    try:
        _assert_semantic_equal(eager, re, rl)
    finally:
        plt.close("all")


def test_b3_gate3_b0_equals_smoke_equals_invariance_ids():
    declared = {r["cell_id"] for r in CELLS}
    smoke = {r["cell_id"] for r in B1.SMOKE_CASES}
    invariance = {r["cell_id"] for r in INVARIANCE_CASES}
    assert declared == smoke == invariance


# ---------------------------------------------------------------------------
# B-4 bounded vector × facet/group seams
# ---------------------------------------------------------------------------

SEAM_CASES = (
    ("selection_vector", "facet_by"),
    ("weights_vector", "facet_by"),
    ("selection_vector", "group_by"),
    ("weights_vector", "group_by"),
)


def _seam_candidate(vector_slot, compose_slot):
    adf = AliasDataFrame(_eager_main_dataframe())
    kwargs = {
        "type": "profile", "bins": 12, "return_data": True,
        "auto_title": False, "vector_compose": "outer",
        vector_slot: list(B1.TARGETS[vector_slot]),
        compose_slot: compose_slot.removesuffix("_by"),
    }
    # compose slot columns are named group/facet, not group_by/facet_by.
    kwargs[compose_slot] = "facet" if compose_slot == "facet_by" else "group"
    return adf.draw("value:x", **kwargs)


def _seam_reference(vector_slot, compose_slot):
    df = _plain_dataframe()
    branches = list(B1.TARGETS[vector_slot])
    channel = "facet" if compose_slot == "facet_by" else "group"
    out = []
    for branch in branches:
        kwargs = {
            "type": "profile", "bins": 12, "return_data": True,
            "auto_title": False, compose_slot: channel,
        }
        if vector_slot == "selection_vector":
            kwargs["selection"] = branch
        else:
            kwargs["weights"] = branch
        out.append(DFDraw(df).draw("value:x", **kwargs))
    return out


def _nonempty_axes(fig):
    return [ax for ax in fig.axes if ax.lines or ax.collections or ax.patches]


@pytest.mark.parametrize("vector_slot,compose_slot", SEAM_CASES,
                         ids=["selection_vector-facet_by", "weights_vector-facet_by",
                              "selection_vector-group_by", "weights_vector-group_by"])
def test_b4_vector_composition_seam(vector_slot, compose_slot):
    cand = _seam_candidate(vector_slot, compose_slot)
    refs = _seam_reference(vector_slot, compose_slot)
    try:
        if compose_slot == "group_by":
            stats = cand[2]
            assert isinstance(stats, list) and len(stats) == 2, (
                f"{vector_slot}×group_by: every vector branch must survive")
            for i, ref in enumerate(refs):
                _assert_frame_semantic_equal(
                    _profile_evidence(stats[i]), _profile_evidence(ref[2]),
                    label=f"{vector_slot}×group_by.branch{i}")
            return

        # Faceting: every non-empty facet must contain every vector branch.
        # The independent reference is one explicit scalar request per branch.
        fig, axes, stats = cand
        assert stats.get("groups") == [0, 1, 2, 3]
        visible_axes = _nonempty_axes(fig)
        assert len(visible_axes) == 4
        for facet_index, ax in enumerate(visible_axes):
            assert len(ax.lines) >= 2, (
                f"{vector_slot}×facet_by facet={facet_index}: expected both vector "
                f"branches, observed {len(ax.lines)} profile line(s)")
    finally:
        plt.close("all")
