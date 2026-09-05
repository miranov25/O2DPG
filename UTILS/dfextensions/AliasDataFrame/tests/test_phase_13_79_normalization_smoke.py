"""PHASE_13_79_ADF BN-1 — normalization integration SMOKE contract.

Ratified BN-0 core:
    normalize='ratio' × 3 curve sources × 5 operand forms × eager/lazy,
    with vector_compose='outer' pinned explicitly.

This checkpoint is test-first.  No ADF/dfdraw production code is changed.
Known implementation gaps remain strict XFAIL with exact signature checks.
"""
from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tests import test_phase_13_79_slot_grid as B1


CONTRACT_PATH = Path(__file__).with_name("phase_13_79_normalization_contract.json")
CONTRACT = json.loads(CONTRACT_PATH.read_text())
CELLS = tuple(CONTRACT["cells"])
FORMS = tuple(CONTRACT["axes"]["forms"])
SOURCES = tuple(CONTRACT["axes"]["curve_sources"])
MODES = tuple(CONTRACT["axes"]["loading_modes"])

BUG_VECTOR_QUALIFIED = "BUG_20260701_ADF_subframe_ref_slot_symmetry"
BUG_Y_VECTOR_ROUTING = "BUG_dfdraw_20260905_draw_profile_y_vector_normalize_ignored"


def _expected_ids():
    return {
        f"normalize:ratio:{source}:{form}:{mode}"
        for source, form, mode in product(SOURCES, FORMS, MODES)
    }


def _validate_contract():
    ids = [row["cell_id"] for row in CELLS]
    assert len(ids) == len(set(ids)), "duplicate BN-0 normalization cell id"
    assert set(ids) == _expected_ids(), (
        "BN-0 core drift: "
        f"missing={sorted(_expected_ids() - set(ids))} "
        f"extra={sorted(set(ids) - _expected_ids())}"
    )
    assert CONTRACT["axes"]["canonical_mode"] == "ratio"
    assert CONTRACT["axes"]["vector_compose"] == "outer"
    assert set(CONTRACT["governing_global_decisions"]) == {
        "AD-4/13.76.ADF", "AD-16/13.76.ADF"
    }
    assert set(CONTRACT["local_ratified_decisions"]) == {
        "BN0-D1", "BN0-D2", "BN0-D3", "BN0-D4", "BN0-D5"
    }
    assert all(row["interface_contract"] == "SUPPORTED" for row in CELLS)
    assert {row["current_state"] for row in CELLS} <= {"PASSING", "KNOWN_GAP"}


def test_bn0_contract_complete_and_ratified():
    _validate_contract()


def test_bn0_contract_measured_state_arithmetic():
    passing = [r for r in CELLS if r["current_state"] == "PASSING"]
    gaps = [r for r in CELLS if r["current_state"] == "KNOWN_GAP"]
    assert len(passing) == 12
    assert len(gaps) == 18
    by_bug = {}
    for row in gaps:
        by_bug[row["owning_bug"]] = by_bug.get(row["owning_bug"], 0) + 1
        sig = row["current_gap_signature"]
        assert sig and sig["type"] and sig["message_fragment"]
    assert by_bug == {BUG_VECTOR_QUALIFIED: 8, BUG_Y_VECTOR_ROUTING: 10}


def test_bn0_non_profile_decision_is_explicit_refusal_contract():
    rows = CONTRACT["non_profile_contract"]
    assert {r["plot_type"] for r in rows} == {"hist", "scatter", "hist2d"}
    assert all(r["interface_contract"] == "REFUSE_BY_DESIGN" for r in rows)
    assert all(r["required_behavior"].startswith("raise deterministic") for r in rows)


def _cell_id(row):
    return row["cell_id"].split("normalize:ratio:", 1)[1].replace(":", "-")


def _target_for_source(source, form):
    if source == "selection_vector":
        return B1._target_value("selection_vector", form)
    if source == "weights_vector":
        return B1._target_value("weights_vector", form)
    mapping = {
        "column": ["value", "color"],
        "alias": ["value_alias", "color_alias"],
        "expression": ["value+0*aux", "color+0*aux"],
        "subframe": ["S.value", "S.color"],
        "struct": ["st.s_value", "st.s_color"],
    }
    return mapping[form]


def _materialize_eager_aliases(adf, source):
    names = {
        "selection_vector": ["sel0_alias", "sel1_alias"],
        "weights_vector": ["w0_alias", "w1_alias"],
        "y_vector": ["value_alias", "color_alias"],
    }[source]
    adf.materialize_aliases(names=names)


def _make_request(row):
    source = row["curve_source"]
    form = row["operand_form"]
    target = _target_for_source(source, form)
    kwargs = {
        "type": "profile",
        "normalize": "ratio",
        "normalize_layout": "overlay+diff",
        "vector_compose": "outer",
        "bins": 8,
        "auto_title": False,
    }
    if source == "selection_vector":
        expr = "value:x"
        kwargs["selection_vector"] = target
    elif source == "weights_vector":
        expr = "value:x"
        kwargs["weights_vector"] = target
    else:
        expr = "[" + ",".join(target) + "]:x"
    return expr, kwargs


def _assert_normalized_result(row, result):
    assert isinstance(result, tuple) and len(result) >= 3, (
        f"{row['cell_id']}: draw did not return (fig, ax, stats)"
    )
    stats = result[2]
    assert isinstance(stats, dict) and "normalize_data" in stats, (
        "BN-SMOKE expected normalize_data dict for ratio; "
        f"got {type(stats).__name__} for {row['cell_id']}"
    )
    assert stats.get("normalize_mode") == "ratio"
    frame = stats["normalize_data"]
    assert isinstance(frame, pd.DataFrame) and not frame.empty
    required = {"value", "signal_count", "reference_count", "mask_undefined"}
    assert required <= set(frame.columns)
    populated = (frame["signal_count"].to_numpy() > 0) & (frame["reference_count"].to_numpy() > 0)
    assert populated.any(), f"{row['cell_id']}: no jointly populated normalization bin"
    vals = frame.loc[populated, "value"].to_numpy(dtype=float)
    assert np.isfinite(vals).any(), f"{row['cell_id']}: no finite ratio value"


def _assert_gap_signature(row, exc):
    sig = row["current_gap_signature"]
    assert type(exc).__name__ == sig["type"], (
        f"{row['cell_id']}: expected {sig['type']}, got {type(exc).__name__}: {exc}"
    )
    assert sig["message_fragment"] in str(exc), (
        f"{row['cell_id']}: wrong current-gap signature: {exc}"
    )


def _param(row):
    if row["current_state"] == "PASSING":
        return pytest.param(row, id=_cell_id(row))
    return pytest.param(
        row,
        id=_cell_id(row),
        marks=pytest.mark.xfail(strict=True, reason=f"{row['owning_bug']} | {row['current_gap_signature']}"),
    )


PARAMS = tuple(_param(row) for row in CELLS)


@pytest.mark.parametrize("row", PARAMS)
def test_bn_smoke_ratio_core(row):
    # BN-0 uses only base eager/lazy modes.  B1's established fixture builder
    # supplies the same alias/subframe/struct representation machinery used by
    # the systematic draw grid.
    b1_row = {
        "slot": row["curve_source"] if row["curve_source"] != "y_vector" else "expr",
        "form": row["operand_form"],
        "mode": row["loading_mode"],
    }
    adf = B1._make_adf(b1_row, (None, None))
    if row["operand_form"] == "alias" and row["loading_mode"] == "eager":
        _materialize_eager_aliases(adf, row["curve_source"])
    expr, kwargs = _make_request(row)

    figs = []
    try:
        if row["current_state"] == "KNOWN_GAP":
            try:
                result = adf.draw(expr, **kwargs)
                fig = result[0]
                figs.extend(fig if isinstance(fig, list) else [fig])
                _assert_normalized_result(row, result)
            except Exception as exc:
                _assert_gap_signature(row, exc)
                raise
            return  # no exception => strict XPASS and therefore failure

        result = adf.draw(expr, **kwargs)
        fig = result[0]
        figs.extend(fig if isinstance(fig, list) else [fig])
        _assert_normalized_result(row, result)
    finally:
        for fig in figs:
            if fig is not None:
                plt.close(fig)
