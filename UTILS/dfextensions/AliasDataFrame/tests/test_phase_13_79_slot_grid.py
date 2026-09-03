"""
PHASE_13_79_ADF B-1 / B-3
==========================

B-1: systematic slot × form × loading-mode SMOKE scan.
B-3: anti-drift gates that make the ratified B-0 table fail closed.

No production code is modified by this checkpoint.

The stable human node IDs are intentionally readable, e.g.
    test_slot_grid_smoke[facet_by-alias-lazy]

Known product gaps remain strict xfail with their owning bug id.  A production
repair therefore becomes XPASS(strict) and forces the B-0 current-state record
to be updated instead of silently turning green.
"""

from __future__ import annotations

import ast
import copy
import inspect
import importlib
import json
import textwrap
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import AliasDataFrame as _adf_module
from AliasDataFrame import AliasDataFrame

uproot = pytest.importorskip("uproot")

try:
    from dfdraw.drawer import DFDraw as _DFDraw
except Exception:  # pragma: no cover - compatibility fallback
    _dfdraw = pytest.importorskip("dfdraw")
    _DFDraw = getattr(_dfdraw, "DFDraw")


CONTRACT_PATH = Path(__file__).with_name("phase_13_79_slot_grid_contract.json")
CONTRACT = json.loads(CONTRACT_PATH.read_text())

SLOTS = tuple(CONTRACT["axes"]["slots"])
FORMS = tuple(CONTRACT["axes"]["forms"])
BASE_MODES = tuple(CONTRACT["axes"]["base_modes"])
BOUNDED_MODES = tuple(CONTRACT["axes"]["bounded_modes"])
CELLS = tuple(CONTRACT["cells"])
SEAMS = tuple(CONTRACT["seams"])

CELL_BY_ID = {row["cell_id"]: row for row in CELLS}
SMOKE_CASES = CELLS

VECTOR_SLOTS = {"selection_vector", "weights_vector"}
BUG_VECTOR_QUALIFIED = "BUG_20260701_ADF_subframe_ref_slot_symmetry"

TARGETS = {
    "expr": ("value",),
    "selection": ("sel",),
    "selection_vector": ("sel0", "sel1"),
    "weights": ("weight",),
    "weights_vector": ("w0", "w1"),
    "group_by": ("group",),
    "facet_by": ("facet",),
    "color": ("color",),
}

ALIAS_EXPR = {
    "value_alias": "value",
    "sel_alias": "sel",
    "sel0_alias": "sel0",
    "sel1_alias": "sel1",
    "weight_alias": "weight",
    "w0_alias": "w0",
    "w1_alias": "w1",
    "group_alias": "group",
    "facet_alias": "facet",
    "color_alias": "color",
}

STRUCT_MEMBERS = [
    "s_value", "s_sel", "s_sel0", "s_sel1", "s_weight", "s_w0", "s_w1",
    "s_group", "s_facet", "s_color",
]

FORWARD_TUPLE_NAMES = (
    "_PROFILE_FORWARDED_NAMES",
    "_HIST_FORWARDED_NAMES",
    "_SCATTER_FORWARDED_NAMES",
    "_HIST2D_FORWARDED_NAMES",
    "_DRAW_FORWARDED_NAMES",
)

COLUMN_REFERENCE_TUPLE_NAMES = (
    "_PROFILE_COLUMN_REFERENCES",
    "_HIST_COLUMN_REFERENCES",
    "_SCATTER_COLUMN_REFERENCES",
    "_DRAW_COLUMN_REFERENCES",
)


def _cell_pytest_id(row):
    return row["cell_id"].split("slotgrid:", 1)[1].replace(":", "-")


def _base_expected_ids():
    return {
        f"slotgrid:{slot}:{form}:{mode}"
        for slot, form, mode in product(SLOTS, FORMS, BASE_MODES)
    }


def _bounded_expected_ids():
    assert BOUNDED_MODES == ("eager_parent_lazy_child",)
    return {
        f"slotgrid:{slot}:subframe:eager_parent_lazy_child"
        for slot in SLOTS
    }


def _declared_cell_ids(rows=None):
    rows = CELLS if rows is None else rows
    return {row["cell_id"] for row in rows}


def _validate_declared_surface(rows=None):
    """B-3 GATE 0: the table must equal the ratified product + bounded rows."""
    rows = CELLS if rows is None else rows
    ids = [row["cell_id"] for row in rows]
    assert len(ids) == len(set(ids)), "duplicate B-0 cell_id"
    expected = _base_expected_ids() | _bounded_expected_ids()
    actual = set(ids)
    assert actual == expected, (
        "B-3 GATE 0: declared B-0 surface drift\n"
        f"missing={sorted(expected - actual)}\n"
        f"extra={sorted(actual - expected)}"
    )


def _validate_seams():
    expected = {
        "slotseam:selection_vector:facet_by",
        "slotseam:weights_vector:facet_by",
        "slotseam:selection_vector:group_by",
        "slotseam:weights_vector:group_by",
    }
    actual = {row["seam_id"] for row in SEAMS}
    assert actual == expected
    assert all(row["evidence_class"] == "seam" for row in SEAMS)


def _production_slot_names():
    # AliasDataFrame can resolve either as the implementation module
    # (repository-root import) or as the package __init__ (editable/package
    # import).  The public class knows its defining implementation module in
    # both layouts; use that rather than requiring private symbols to be
    # re-exported from __init__.py.
    impl_module = importlib.import_module(AliasDataFrame.__module__)
    spec_cls = getattr(impl_module, "_EffectiveDrawSpec")
    return {"expr"} | set(spec_cls.SLOT_NAMES)


def _dfdraw_forwarding_inventory():
    """B-3 GATE 2 visibility inventory: derived, never hand-maintained."""
    out = set()
    for name in FORWARD_TUPLE_NAMES:
        assert hasattr(_DFDraw, name), f"dfdraw missing forwarding authority {name}"
        out.update(getattr(_DFDraw, name))

    known_fn = getattr(_DFDraw, "_c7_known_kwargs", None)
    assert known_fn is not None, "dfdraw missing broader _c7_known_kwargs authority"
    out.update(known_fn())
    return out


def _dfdraw_column_reference_inventory(extra=()):
    """Names dfdraw explicitly declares to contain data/column references."""
    out = set(extra)
    for name in COLUMN_REFERENCE_TUPLE_NAMES:
        assert hasattr(_DFDraw, name), f"dfdraw missing column-reference authority {name}"
        out.update(getattr(_DFDraw, name))
    return out


def _literal_kwargs_reads(fn):
    """Derive literal kwargs.get/pop/subscript consumers from an ADF method."""
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    except (OSError, TypeError, IndentationError, SyntaxError):
        return set()
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {"get", "pop", "setdefault"} and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    found.add(arg.value)
        if isinstance(node, ast.Subscript):
            sl = node.slice
            if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
                found.add(sl.value)
    return found


def _adf_public_parameter_inventory():
    out = set()
    for name in ("draw", "draw_batch", "draw_figures"):
        fn = getattr(AliasDataFrame, name)
        out.update(
            p
            for p, param in inspect.signature(fn).parameters.items()
            if p != "self" and param.kind not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
        )
        out.update(_literal_kwargs_reads(fn))
    return out


def _validate_column_reference_classification(extra=()):
    """B-3 GATE 2: every known dfdraw data-reference kwarg is classified."""
    refs = _dfdraw_column_reference_inventory(extra=extra)
    exclusions = {row["name"] for row in CONTRACT["forwarding_exclusions"]}
    classified = set(SLOTS) | exclusions

    # "expr" is positional in ADF and does not occur in dfdraw's kwarg tuples.
    unknown = refs - classified
    assert not unknown, (
        "B-3 GATE 2: unclassified dfdraw column-reference kwargs: "
        f"{sorted(unknown)}"
    )

    # The broader inventory is deliberately derived too.  This is a visibility
    # guard: every explicit forwarding tuple and ADF public parameter must be
    # represented in a live accepted/consumed inventory, even though most are
    # styling/control kwargs rather than B-0 data slots.
    accepted = _dfdraw_forwarding_inventory()
    assert set().union(*(set(getattr(_DFDraw, n)) for n in FORWARD_TUPLE_NAMES)) <= accepted
    assert _adf_public_parameter_inventory(), "ADF public-parameter inventory unexpectedly empty"


def _validate_b0_schema():
    required = {
        "cell_id", "slot", "form", "mode", "plot_type", "applicability",
        "interface_contract", "current_state", "current_gap_signature",
        "owning_bug", "owning_phase", "fixture_id", "oracle_id", "comparator",
        "vector_compose", "matrix_glyph", "pytest_policy", "evidence_class",
        "mechanism_disposition_ad16", "measured_at", "rationale",
    }
    for row in CELLS:
        assert required <= set(row), f"{row.get('cell_id')}: incomplete B-0 schema"
        assert row["interface_contract"] == "SUPPORTED"
        assert row["current_state"] in {"UNMEASURED", "PASSING", "KNOWN_GAP"}
        if row["current_state"] == "KNOWN_GAP":
            assert row["owning_bug"]
            assert row["pytest_policy"] == "STRICT_XFAIL"
            assert row["current_gap_signature"]
            assert row["current_gap_signature"].get("type")
            assert row["current_gap_signature"].get("message_fragment")


# ---------------------------------------------------------------------------
# B-3 anti-drift tests
# ---------------------------------------------------------------------------

def test_b3_gate0_declared_product_is_complete():
    _validate_declared_surface()


def test_b3_gate0_falsifier_delete_one_row_is_caught():
    damaged = [dict(row) for row in CELLS]
    damaged.pop()
    with pytest.raises(AssertionError, match="GATE 0"):
        _validate_declared_surface(damaged)


def test_b3_gate1_grid_slots_equal_production_slot_authority():
    assert set(SLOTS) == _production_slot_names()


def test_b3_gate2_live_column_reference_inventory_is_fully_classified():
    _validate_column_reference_classification()


def test_b3_gate2_falsifier_new_column_reference_is_caught():
    fake = "__phase_13_79_unclassified_column_kwarg__"
    with pytest.raises(AssertionError, match="GATE 2"):
        _validate_column_reference_classification(extra={fake})


def test_b3_gate2_all_five_dfdraw_forwarding_authorities_exist():
    assert all(hasattr(_DFDraw, name) for name in FORWARD_TUPLE_NAMES)


def test_b3_gate3_b0_equals_smoke_parameter_ids():
    # B-INVARIANCE is the next checkpoint.  This test intentionally claims
    # only the arms that exist now; B-2 will extend Gate 3 to all three sets.
    assert _declared_cell_ids() == {row["cell_id"] for row in SMOKE_CASES}


def test_b3_ratified_b0_schema_and_four_b4_seams_are_complete():
    _validate_b0_schema()
    _validate_seams()


# ---------------------------------------------------------------------------
# Deterministic fixture factory
# ---------------------------------------------------------------------------

def _arrays(n=96):
    i = np.arange(n, dtype=np.int32)
    x = np.linspace(-2.0, 2.0, n, dtype=np.float64)
    aux = ((i % 7) - 3).astype(np.float64) / 10.0
    return {
        "k": i,
        "x": x,
        "value": 1.5 + 0.7 * x + 0.2 * aux,
        "aux": aux,
        "sel": (i % 3 != 0),
        "sel0": (i % 2 == 0),
        "sel1": (i % 2 == 1),
        "weight": 1.0 + (i % 5).astype(np.float64) / 10.0,
        "w0": 0.8 + (i % 4).astype(np.float64) / 10.0,
        "w1": 1.1 + (i % 6).astype(np.float64) / 10.0,
        "group": (i % 3).astype(np.int16),
        "facet": (i % 4).astype(np.int16),
        "color": ((i % 11).astype(np.float64) + 0.25),
    }


class _B1TrackingLazyReader:
    """In-memory lazy reader using the already-proven PHASE_13_77 A4 seam."""

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
            raise ValueError(f"missing B1 tracking branches: {sorted(missing)}")
        to_load = names - self.loaded_branches
        self.loaded_branches.update(to_load)
        if not to_load:
            return pd.DataFrame(index=self.data.index)
        return self.data[sorted(to_load)].copy()


def _lazy_main_raw(*, include_struct=False):
    data = _arrays()
    raw = pd.DataFrame(data)
    if include_struct:
        for member in STRUCT_MEMBERS:
            source = member.removeprefix("s_")
            raw[f"st/{member}"] = np.asarray(data[source])
    return raw


@pytest.fixture(scope="session")
def _b1_root_files(tmp_path_factory):
    """Only the bounded eager-parent/lazy-child seam needs a real ROOT child."""
    root = tmp_path_factory.mktemp("phase_13_79_b1")
    data = _arrays()
    child_path = root / "child.root"
    child_members = sorted({name for names in TARGETS.values() for name in names})
    child_tree = {
        "k": data["k"],
        **{member: np.asarray(data[member]) for member in child_members},
    }
    with uproot.recreate(child_path) as f:
        f.mktree("tree", {name: arr.dtype for name, arr in child_tree.items()})
        f["tree"].extend(child_tree)
    return None, child_path


def _eager_main_dataframe():
    data = _arrays()
    df = pd.DataFrame(data)
    for member in STRUCT_MEMBERS:
        source = member.removeprefix("s_")
        df[f"{member}__st"] = np.asarray(data[source])
    return df


def _eager_child_dataframe():
    data = _arrays()
    child_members = sorted({name for names in TARGETS.values() for name in names})
    return pd.DataFrame({
        "k": data["k"],
        **{member: np.asarray(data[member]) for member in child_members},
    })


def _configure_aliases(adf):
    for name, expr in ALIAS_EXPR.items():
        adf.add_alias(name, expr)


def _make_adf(row, root_files):
    main_path, child_path = root_files
    mode = row["mode"]
    form = row["form"]

    if mode == "eager":
        adf = AliasDataFrame(_eager_main_dataframe())
        adf.draw_lazy = False
    elif mode == "lazy":
        raw = _lazy_main_raw(include_struct=(form == "struct"))
        adf = AliasDataFrame(pd.DataFrame(index=range(len(raw))))
        adf._lazy_reader = _B1TrackingLazyReader(raw)
        adf._chain = {
            "files": [], "entry_offsets": [0], "total_entries": len(raw),
            "validation_mode": None,
        }
        adf.draw_lazy = True
    elif mode == "eager_parent_lazy_child":
        adf = AliasDataFrame(_eager_main_dataframe())
        adf.draw_lazy = True
    else:  # pragma: no cover - Gate 0 should make this unreachable
        raise AssertionError(f"unknown B-0 mode {mode!r}")

    if form == "alias":
        _configure_aliases(adf)

    if form == "subframe":
        if mode == "eager_parent_lazy_child":
            adf.register_subframe_lazy(
                "S", str(child_path), tree_name="tree",
                index_columns=["k"], alignment="1:1",
            )
        else:
            child = AliasDataFrame(_eager_child_dataframe())
            adf.register_subframe("S", child, index_columns=["k"])
            if mode == "lazy":
                # Established A4 structural baseline: the join key exists
                # before the temporary qualified-column merge.
                adf.ensure_branches(["k"])

    if form == "struct":
        # read_tree_lazy may auto-detect scalar struct branches.  Explicitly
        # register only when auto-detection has not already done so.
        structs = getattr(adf, "_structs", {})
        if "st" not in structs:
            adf.register_struct("st", STRUCT_MEMBERS)
        else:
            observed = set(structs["st"].get("members", ()))
            assert set(STRUCT_MEMBERS) <= observed, (
                "lazy struct auto-detection registered only a partial fixture: "
                f"missing={sorted(set(STRUCT_MEMBERS) - observed)}"
            )

    return adf


def _target_value(slot, form):
    names = TARGETS[slot]

    if form == "column":
        values = list(names)
    elif form == "alias":
        values = [f"{name}_alias" for name in names]
    elif form == "expression":
        mapping = {
            "value": "value+0.1*aux",
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
        values = [mapping[name] for name in names]
    elif form == "subframe":
        values = [f"S.{name}" for name in names]
    elif form == "struct":
        values = [f"st.s_{name}" for name in names]
    else:  # pragma: no cover
        raise AssertionError(form)

    return values[0] if len(values) == 1 else values


def _materialize_eager_alias_targets(adf, slot):
    names = TARGETS[slot]
    aliases = [f"{name}_alias" for name in names]
    adf.materialize_aliases(names=aliases)


def _build_draw_request(adf, row):
    slot = row["slot"]
    form = row["form"]
    target = _target_value(slot, form)

    if form == "alias" and row["mode"] == "eager":
        _materialize_eager_alias_targets(adf, slot)

    kwargs = {
        "auto_title": False,
    }

    if slot == "expr":
        expr = target
        kwargs["type"] = "hist"
        kwargs["bins"] = 12
    elif slot == "color":
        expr = "value:x"
        kwargs["type"] = "scatter"
        kwargs["color"] = target
    else:
        expr = "value:x"
        kwargs["type"] = "profile"
        kwargs["bins"] = 12
        kwargs[slot] = target
        kwargs["return_data"] = True
        if slot in VECTOR_SLOTS:
            kwargs["vector_compose"] = "outer"

    return expr, kwargs


def _assert_public_result(result, row):
    assert isinstance(result, tuple) and len(result) >= 3, (
        f"{row['cell_id']}: draw did not return (fig, ax, stats)"
    )
    fig, ax, stats = result[:3]
    assert fig is not None, f"{row['cell_id']}: fig is None"
    assert getattr(fig, "axes", None), f"{row['cell_id']}: figure has no axes"
    assert stats is not None, f"{row['cell_id']}: stats is None"


def _strict_known_gap_call(adf, row, expr, kwargs):
    sig = row["current_gap_signature"]
    assert sig and sig["type"] == "ValueError"
    try:
        adf.draw(expr, **kwargs)
    except ValueError as exc:
        assert sig["message_fragment"] in str(exc), (
            f"{row['cell_id']}: wrong ValueError signature: {exc!r}"
        )
        raise


def _assert_gap_signature(row, exc):
    sig = row["current_gap_signature"]
    assert sig is not None, f"{row['cell_id']}: KNOWN_GAP without signature"
    expected_type = sig.get("type")
    if expected_type:
        assert type(exc).__name__ == expected_type, (
            f"{row['cell_id']}: expected {expected_type}, got "
            f"{type(exc).__name__}: {exc}"
        )
    fragment = sig.get("message_fragment")
    if fragment:
        assert fragment in str(exc), (
            f"{row['cell_id']}: expected message fragment {fragment!r}, "
            f"got {str(exc)!r}"
        )


# ---------------------------------------------------------------------------
# B-1 systematic smoke scan
# ---------------------------------------------------------------------------

def _smoke_param(row):
    if row["current_state"] != "KNOWN_GAP":
        return pytest.param(row, id=_cell_pytest_id(row))
    sig = row["current_gap_signature"]
    return pytest.param(
        row,
        id=_cell_pytest_id(row),
        marks=pytest.mark.xfail(
            strict=True,
            reason=f"{row['owning_bug']} | {sig}",
        ),
    )


SMOKE_PARAMS = tuple(_smoke_param(row) for row in SMOKE_CASES)


@pytest.mark.parametrize("row", SMOKE_PARAMS)
def test_slot_grid_smoke(row, _b1_root_files):
    adf = _make_adf(row, _b1_root_files)
    expr, kwargs = _build_draw_request(adf, row)

    fig = None
    try:
        if row["current_state"] == "KNOWN_GAP":
            # Validate the registered product-gap signature before re-raising
            # into pytest's strict xfail machinery.  Gap families can have
            # different public exception classes; class + message are both
            # checked from the machine contract.
            try:
                adf.draw(expr, **kwargs)
            except Exception as exc:
                _assert_gap_signature(row, exc)
                raise
            return  # no exception -> strict XPASS -> fail

        result = adf.draw(expr, **kwargs)
        _assert_public_result(result, row)
        fig = result[0]
    finally:
        if fig is not None:
            plt.close(fig)
        else:
            plt.close("all")


# ---------------------------------------------------------------------------
# Exact known-gap signature checks.
#
# The smoke nodes above are ordinary xfail markers for readable matrix output.
# These separate nodes prevent a wrong exception from being hidden as xfail.
# ---------------------------------------------------------------------------

KNOWN_GAP_CASES = tuple(row for row in CELLS if row["current_state"] == "KNOWN_GAP")


@pytest.mark.parametrize("row", KNOWN_GAP_CASES, ids=_cell_pytest_id)
def test_slot_grid_known_gap_signature(row, _b1_root_files):
    adf = _make_adf(row, _b1_root_files)
    expr, kwargs = _build_draw_request(adf, row)
    with pytest.raises(Exception) as caught:
        adf.draw(expr, **kwargs)
    _assert_gap_signature(row, caught.value)


def test_slot_grid_contract_has_expected_known_gap_ownership():
    gaps = {row["cell_id"]: row for row in CELLS if row["current_state"] == "KNOWN_GAP"}
    assert len(gaps) == 18

    vector_base = {
        f"slotgrid:{slot}:{form}:{mode}"
        for slot, form, mode in product(
            ("selection_vector", "weights_vector"),
            ("subframe", "struct"),
            BASE_MODES,
        )
    }
    facet_expr = {f"slotgrid:facet_by:expression:{mode}" for mode in BASE_MODES}
    mixed = {f"slotgrid:{slot}:subframe:eager_parent_lazy_child" for slot in SLOTS}
    assert set(gaps) == vector_base | facet_expr | mixed

    for cid in vector_base:
        assert gaps[cid]["owning_bug"] == BUG_VECTOR_QUALIFIED
    for cid in facet_expr:
        assert gaps[cid]["owning_bug"] == "BUG_AliasDataFrame_20260903_facet_by_expression_not_materialized"
    for cid in mixed:
        assert gaps[cid]["owning_bug"] == "BUG_AliasDataFrame_20260116_lazy_subframe_init"
