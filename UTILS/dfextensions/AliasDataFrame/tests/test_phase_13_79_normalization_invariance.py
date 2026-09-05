"""PHASE_13_79_ADF BN-2 — normalization numerical invariance.

Strengthens BN-1 SMOKE with independent raw-fixture numerical oracles.
No dfdraw normalization helper is used to build expected values.

Coverage:
  * 30 ratified ratio core rows (12 numerical PASS + 18 strict current gaps),
  * 6 explicit eager/lazy equivalence pairs for the currently reachable core,
  * 5 bounded mode seams: delta/log_ratio/pull/callable-2/callable-4.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tests import test_phase_13_79_normalization_smoke as BN1
from tests import test_phase_13_79_slot_grid as B1


CONTRACT_PATH = Path(__file__).with_name("phase_13_79_normalization_contract.json")
CONTRACT = json.loads(CONTRACT_PATH.read_text())
CELLS = tuple(CONTRACT["cells"])
BINS = 8


def _bin_indices(x, bins=BINS):
    x = np.asarray(x, dtype=float)
    lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    edges = np.linspace(lo, hi, bins + 1)
    idx = np.digitize(x, edges) - 1
    idx[x == edges[-1]] = bins - 1
    inside = (idx >= 0) & (idx < bins)
    return idx, inside, edges


def _profile_oracle(*, mask=None, weights=None, bins=BINS):
    """Independent per-bin profile oracle from the deterministic raw fixture."""
    a = B1._arrays()
    x = np.asarray(a["x"], dtype=float)
    y = np.asarray(a["value"], dtype=float)
    idx, inside, edges = _bin_indices(x, bins=bins)
    if mask is None:
        mask = np.ones(len(x), dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
    if weights is not None:
        weights = np.asarray(weights, dtype=float)

    centers = (edges[:-1] + edges[1:]) / 2.0
    means = np.full(bins, np.nan, dtype=float)
    sigmas = np.full(bins, np.nan, dtype=float)
    counts = np.zeros(bins, dtype=int)

    for b in range(bins):
        take = inside & mask & (idx == b)
        yy = y[take]
        counts[b] = len(yy)
        if not len(yy):
            continue
        if weights is None:
            means[b] = float(np.mean(yy))
            if len(yy) > 1:
                sigmas[b] = float(np.std(yy, ddof=1))
        else:
            ww = weights[take]
            sum_w = float(np.sum(ww))
            if sum_w <= 0:
                continue
            means[b] = float(np.sum(ww * yy) / sum_w)
            if len(yy) > 1:
                sigmas[b] = float(np.sqrt(np.sum(ww * (yy - means[b]) ** 2) / sum_w))

    return {
        "x_center": centers,
        "central": means,
        "sigma": sigmas,
        "count": counts,
    }


def _ratio_core_oracle(source):
    a = B1._arrays()
    if source == "selection_vector":
        signal = _profile_oracle(mask=a["sel0"])
        reference = _profile_oracle(mask=a["sel1"])
    elif source == "weights_vector":
        signal = _profile_oracle(weights=a["w0"])
        reference = _profile_oracle(weights=a["w1"])
    else:
        raise AssertionError(f"no reachable BN-2 raw oracle for source {source!r}")

    s = signal["central"]
    r = reference["central"]
    sc = signal["count"]
    rc = reference["count"]
    mask = (sc < 1) | (rc < 1) | (r == 0) | ~np.isfinite(r)
    with np.errstate(divide="ignore", invalid="ignore"):
        value = s / r
    value = value.astype(float)
    value[mask] = np.nan
    return signal, reference, value, mask


def _assert_ratio_core_oracle(row, result):
    assert isinstance(result, tuple) and len(result) >= 3
    stats = result[2]
    assert isinstance(stats, dict) and "normalize_data" in stats
    nd = stats["normalize_data"]
    assert isinstance(nd, pd.DataFrame)

    signal, reference, expected, mask = _ratio_core_oracle(row["curve_source"])
    np.testing.assert_allclose(nd["x_center"].to_numpy(float), signal["x_center"], rtol=0, atol=1e-14)
    np.testing.assert_array_equal(nd["signal_count"].to_numpy(int), signal["count"])
    np.testing.assert_array_equal(nd["reference_count"].to_numpy(int), reference["count"])
    np.testing.assert_allclose(nd["signal_central"].to_numpy(float), signal["central"], rtol=1e-13, atol=1e-13, equal_nan=True)
    np.testing.assert_allclose(nd["reference_central"].to_numpy(float), reference["central"], rtol=1e-13, atol=1e-13, equal_nan=True)
    np.testing.assert_array_equal(nd["mask_undefined"].to_numpy(bool), mask)
    np.testing.assert_allclose(nd["value"].to_numpy(float), expected, rtol=1e-13, atol=1e-13, equal_nan=True)


def _build_adf_for_row(row):
    b1_row = {
        "slot": row["curve_source"] if row["curve_source"] != "y_vector" else "expr",
        "form": row["operand_form"],
        "mode": row["loading_mode"],
    }
    adf = B1._make_adf(b1_row, (None, None))
    if row["operand_form"] == "alias" and row["loading_mode"] == "eager":
        BN1._materialize_eager_aliases(adf, row["curve_source"])
    return adf


def _param(row):
    test_id = row["cell_id"].split("normalize:ratio:", 1)[1].replace(":", "-")
    if row["current_state"] == "PASSING":
        return pytest.param(row, id=test_id)
    return pytest.param(
        row,
        id=test_id,
        marks=pytest.mark.xfail(
            strict=True,
            reason=f"{row['owning_bug']} | {row['current_gap_signature']}",
        ),
    )


@pytest.mark.parametrize("row", tuple(_param(row) for row in CELLS))
def test_bn2_ratio_core_independent_numerical_oracle(row):
    adf = _build_adf_for_row(row)
    expr, kwargs = BN1._make_request(row)
    figs = []
    try:
        if row["current_state"] == "KNOWN_GAP":
            try:
                result = adf.draw(expr, **kwargs)
                fig = result[0]
                figs.extend(fig if isinstance(fig, list) else [fig])
                # Preserve the exact BN-1 current-gap signature first.  If the
                # routing gap has disappeared, this returns normally and the
                # independent oracle below decides whether the repaired row is
                # numerically correct before strict XPASS forces reconciliation.
                BN1._assert_normalized_result(row, result)
                _assert_ratio_core_oracle(row, result)
            except Exception as exc:
                BN1._assert_gap_signature(row, exc)
                raise
            return

        result = adf.draw(expr, **kwargs)
        fig = result[0]
        figs.extend(fig if isinstance(fig, list) else [fig])
        _assert_ratio_core_oracle(row, result)
    finally:
        for fig in figs:
            if fig is not None:
                plt.close(fig)


PAIR_CASES = tuple(
    (source, form)
    for source in ("selection_vector", "weights_vector")
    for form in ("column", "alias", "expression")
)


def _run_core(source, form, loading):
    row = next(
        r for r in CELLS
        if r["curve_source"] == source
        and r["operand_form"] == form
        and r["loading_mode"] == loading
    )
    assert row["current_state"] == "PASSING"
    adf = _build_adf_for_row(row)
    expr, kwargs = BN1._make_request(row)
    result = adf.draw(expr, **kwargs)
    return result


@pytest.mark.parametrize("source,form", PAIR_CASES, ids=lambda x: str(x))
def test_bn2_ratio_eager_lazy_equivalence(source, form):
    eager = _run_core(source, form, "eager")
    lazy = _run_core(source, form, "lazy")
    try:
        e = eager[2]["normalize_data"]
        l = lazy[2]["normalize_data"]
        assert list(e.columns) == list(l.columns)
        for col in (
            "x_center", "value", "signal_central", "reference_central",
            "signal_sigma", "reference_sigma", "error",
        ):
            np.testing.assert_allclose(
                e[col].to_numpy(float), l[col].to_numpy(float),
                rtol=1e-13, atol=1e-13, equal_nan=True,
            )
        for col in ("signal_count", "reference_count", "mask_undefined"):
            np.testing.assert_array_equal(e[col].to_numpy(), l[col].to_numpy())
    finally:
        plt.close(eager[0])
        plt.close(lazy[0])


def _two_curve_raw_stats():
    a = B1._arrays()
    return _profile_oracle(mask=a["sel0"]), _profile_oracle(mask=a["sel1"])


def _run_mode(mode, selection_vector=None):
    adf = B1._make_adf({"slot": "selection_vector", "form": "column", "mode": "eager"}, (None, None))
    kwargs = {
        "type": "profile",
        "selection_vector": selection_vector or ["sel0", "sel1"],
        "normalize": mode,
        "normalize_layout": "overlay+diff",
        "vector_compose": "outer",
        "bins": BINS,
        "auto_title": False,
    }
    return adf.draw("value:x", **kwargs)


def _assert_mode_values(result, expected, mode_name):
    stats = result[2]
    assert stats["normalize_mode"] == mode_name
    actual = stats["normalize_data"]["value"].to_numpy(float)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True)


@pytest.mark.parametrize("mode", ["delta", "log_ratio", "pull"])
def test_bn2_builtin_mode_seams_independent_oracle(mode):
    sig, ref = _two_curve_raw_stats()
    s, r = sig["central"], ref["central"]
    sc, rc = sig["count"], ref["count"]
    with np.errstate(divide="ignore", invalid="ignore"):
        if mode == "delta":
            expected = s - r
        elif mode == "log_ratio":
            expected = np.log(s / r)
        else:
            sem2_s = (sig["sigma"] ** 2) / np.where(sc > 0, sc, 1)
            sem2_r = (ref["sigma"] ** 2) / np.where(rc > 0, rc, 1)
            denom = np.sqrt(sem2_s + sem2_r)
            expected = (s - r) / denom
    undefined = (sc < 1) | (rc < 1)
    if mode == "log_ratio":
        undefined |= (s <= 0) | (r <= 0)
    if mode == "pull":
        undefined |= ~np.isfinite(expected)
    expected = np.asarray(expected, dtype=float)
    expected[undefined] = np.nan

    result = _run_mode(mode)
    try:
        _assert_mode_values(result, expected, mode)
    finally:
        plt.close(result[0])


def _callable_2curve(S):
    return 1.25 * S[0].value - 0.75 * S[1].value


def test_bn2_callable_2curve_seam_independent_oracle():
    sig, ref = _two_curve_raw_stats()
    expected = 1.25 * sig["central"] - 0.75 * ref["central"]
    result = _run_mode(_callable_2curve)
    try:
        _assert_mode_values(result, expected, "callable")
    finally:
        plt.close(result[0])


def _callable_4curve(S):
    with np.errstate(divide="ignore", invalid="ignore"):
        return (S[0].value / S[1].value) / (S[2].value / S[3].value)


def test_bn2_callable_4curve_double_ratio_seam_independent_oracle():
    a = B1._arrays()
    stats = [_profile_oracle(mask=(a["k"] % 4 == i)) for i in range(4)]
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = (stats[0]["central"] / stats[1]["central"]) / (stats[2]["central"] / stats[3]["central"])
    undefined = np.zeros(BINS, dtype=bool)
    for st in stats:
        undefined |= st["count"] < 1
    undefined |= ~np.isfinite(expected)
    expected = np.asarray(expected, dtype=float)
    expected[undefined] = np.nan

    result = _run_mode(
        _callable_4curve,
        selection_vector=["k%4==0", "k%4==1", "k%4==2", "k%4==3"],
    )
    try:
        _assert_mode_values(result, expected, "callable")
    finally:
        plt.close(result[0])
