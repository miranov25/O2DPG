"""PHASE_13_79_ADF BN-3 — bounded normalization integration seams.

Completes the BN-0 bounded integration surface without changing ADF/dfdraw
production code:
  * normalize × group_by / facet_by,
  * draw / draw_batch / draw_figures public-surface forwarding,
  * normalize_layout='diff_only',
  * ADF vector_compose auto-force for selection/weights vectors,
  * non-profile REFUSE_BY_DESIGN executable contract (current warn+ignore gap).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dfextensions.dfdraw import DFDraw
from tests import test_phase_13_79_slot_grid as B1
from tests import test_phase_13_79_normalization_smoke as BN1
from tests import test_phase_13_79_normalization_invariance as BN2


CONTRACT_PATH = Path(__file__).with_name("phase_13_79_normalization_contract.json")
CONTRACT = json.loads(CONTRACT_PATH.read_text())
BINS = 8
BUG_NONPROFILE = "BUG_dfdraw_20260905_nonprofile_normalize_warn_ignore"
CURRENT_NONPROFILE_ASSERT = "BN3 expected REFUSE_BY_DESIGN raise; observed warn+ignore"


def _make_eager_adf():
    return B1._make_adf({"slot": "selection_vector", "form": "column", "mode": "eager"}, (None, None))


def _canonical_kwargs(*, layout="overlay+diff", vector_compose=True):
    kw = {
        "type": "profile",
        "selection_vector": ["sel0", "sel1"],
        "normalize": "ratio",
        "normalize_layout": layout,
        "bins": BINS,
        "auto_title": False,
    }
    if vector_compose:
        kw["vector_compose"] = "outer"
    return kw


def _expected_ratio_for_mask(extra_mask=None):
    a = B1._arrays()
    if extra_mask is None:
        extra_mask = np.ones(len(a["x"]), dtype=bool)
    extra_mask = np.asarray(extra_mask, dtype=bool)
    signal = BN2._profile_oracle(mask=extra_mask & np.asarray(a["sel0"], dtype=bool))
    reference = BN2._profile_oracle(mask=extra_mask & np.asarray(a["sel1"], dtype=bool))
    s = signal["central"]
    r = reference["central"]
    sc = signal["count"]
    rc = reference["count"]
    undefined = (sc < 1) | (rc < 1) | (r == 0) | ~np.isfinite(r)
    with np.errstate(divide="ignore", invalid="ignore"):
        value = s / r
    value = np.asarray(value, dtype=float)
    value[undefined] = np.nan
    return signal, reference, value, undefined


def _assert_frame_matches_independent_ratio(frame, *, source="selection_vector"):
    signal, reference, expected, undefined = BN2._ratio_core_oracle(source)
    np.testing.assert_allclose(frame["x_center"].to_numpy(float), signal["x_center"], rtol=0, atol=1e-14)
    np.testing.assert_array_equal(frame["signal_count"].to_numpy(int), signal["count"])
    np.testing.assert_array_equal(frame["reference_count"].to_numpy(int), reference["count"])
    np.testing.assert_allclose(frame["signal_central"].to_numpy(float), signal["central"], rtol=1e-13, atol=1e-13, equal_nan=True)
    np.testing.assert_allclose(frame["reference_central"].to_numpy(float), reference["central"], rtol=1e-13, atol=1e-13, equal_nan=True)
    np.testing.assert_array_equal(frame["mask_undefined"].to_numpy(bool), undefined)
    np.testing.assert_allclose(frame["value"].to_numpy(float), expected, rtol=1e-13, atol=1e-13, equal_nan=True)


def _close_result_figures(result):
    if result is None:
        return
    try:
        fig = result[0] if isinstance(result, tuple) else None
        if fig is not None:
            for item in (fig if isinstance(fig, list) else [fig]):
                if item is not None:
                    plt.close(item)
    except Exception:
        pass


def test_bn3_group_by_ratio_independent_partition_oracle():
    adf = _make_eager_adf()
    result = adf.draw("value:x", **_canonical_kwargs(), group_by="group")
    try:
        stats = result[2]
        assert stats["normalize_mode"] == "ratio"
        assert stats["group_by"] == "group"
        grouped = stats["normalize_data_grouped"]
        assert list(grouped) == ["0", "1", "2"]
        a = B1._arrays()
        for g in (0, 1, 2):
            signal, reference, expected, undefined = _expected_ratio_for_mask(a["group"] == g)
            got = grouped[str(g)]
            np.testing.assert_allclose(got["bin_centers"], signal["x_center"], rtol=0, atol=1e-14)
            np.testing.assert_array_equal(got["signal_count"], signal["count"])
            np.testing.assert_array_equal(got["reference_count"], reference["count"])
            np.testing.assert_allclose(got["signal_central"], signal["central"], rtol=1e-13, atol=1e-13, equal_nan=True)
            np.testing.assert_allclose(got["reference_central"], reference["central"], rtol=1e-13, atol=1e-13, equal_nan=True)
            np.testing.assert_array_equal(got["mask_undefined"], undefined)
            np.testing.assert_allclose(got["values"], expected, rtol=1e-13, atol=1e-13, equal_nan=True)
    finally:
        _close_result_figures(result)


def test_bn3_facet_by_ratio_independent_partition_oracle():
    # Use group as the facet variable deliberately.  The fixture's `facet`
    # parity is correlated with sel0/sel1 and would make every facet one-sided,
    # which is a degenerate oracle rather than a useful composition test.
    adf = _make_eager_adf()
    result = adf.draw("value:x", **_canonical_kwargs(), facet_by="group")
    try:
        stats = result[2]
        assert stats["normalize_mode"] == "ratio"
        assert stats["facet_by"] == "group"
        faceted = stats["normalize_data_faceted"]
        assert list(faceted) == ["0", "1", "2"]
        a = B1._arrays()
        for g in (0, 1, 2):
            signal, _reference, expected, undefined = _expected_ratio_for_mask(a["group"] == g)
            got = faceted[str(g)]
            np.testing.assert_allclose(got["bin_centers"], signal["x_center"], rtol=0, atol=1e-14)
            np.testing.assert_array_equal(got["mask_undefined"], undefined)
            np.testing.assert_allclose(got["values"], expected, rtol=1e-13, atol=1e-13, equal_nan=True)
    finally:
        _close_result_figures(result)


def _run_surface(surface):
    adf = _make_eager_adf()
    spec = {"expr": "value:x", **_canonical_kwargs()}
    if surface == "draw":
        expr = spec.pop("expr")
        result = adf.draw(expr, **spec)
        return result, result[2]["normalize_data"]
    if surface == "draw_batch":
        result = adf.draw_batch({"norm": spec}, verbose=False)
        return result, result["norm"]["stats"]["normalize_data"]
    if surface == "draw_figures":
        result = adf.draw_figures([{"name": "normfig", "plots": [spec]}], verbose=False)
        return result, result["normfig"]["stats"][0]["normalize_data"]
    raise AssertionError(surface)


@pytest.mark.parametrize("surface", ["draw", "draw_batch", "draw_figures"])
def test_bn3_public_surface_ratio_independent_oracle(surface):
    result, frame = _run_surface(surface)
    try:
        _assert_frame_matches_independent_ratio(frame)
    finally:
        plt.close("all")


def test_bn3_diff_only_layout_preserves_ratio_and_axis_contract():
    adf = _make_eager_adf()
    result = adf.draw("value:x", **_canonical_kwargs(layout="diff_only"))
    try:
        fig, _ax, stats = result
        assert stats["normalize_layout"] == "diff_only"
        assert len(fig.axes) == 1
        _assert_frame_matches_independent_ratio(stats["normalize_data"])
    finally:
        _close_result_figures(result)


@pytest.mark.parametrize("source", ["selection_vector", "weights_vector"])
def test_bn3_vector_compose_autoforce_public_bridge(source, monkeypatch):
    captured = []
    original = DFDraw.draw

    def _spy(self, expr, *args, **kwargs):
        captured.append(kwargs.get("vector_compose"))
        return original(self, expr, *args, **kwargs)

    monkeypatch.setattr(DFDraw, "draw", _spy)
    adf = _make_eager_adf()
    kwargs = {
        "type": "profile",
        "normalize": "ratio",
        "normalize_layout": "overlay+diff",
        "bins": BINS,
        "auto_title": False,
    }
    if source == "selection_vector":
        kwargs["selection_vector"] = ["sel0", "sel1"]
    else:
        kwargs["weights_vector"] = ["w0", "w1"]

    # Deliberately omit vector_compose: ADF must supply outer before delegation.
    result = adf.draw("value:x", **kwargs)
    try:
        assert captured and captured[-1] == "outer"
        _assert_frame_matches_independent_ratio(result[2]["normalize_data"], source=source)
    finally:
        _close_result_figures(result)


NONPROFILE = tuple(CONTRACT["non_profile_contract"])


def _nonprofile_param(row):
    return pytest.param(
        row,
        id=row["plot_type"],
        marks=pytest.mark.xfail(
            strict=True,
            raises=AssertionError,
            reason=f"{row['owning_bug']} | {row['current_gap_signature']}",
        ),
    )


@pytest.mark.parametrize("row", tuple(_nonprofile_param(r) for r in NONPROFILE))
def test_bn3_nonprofile_normalize_refuse_by_design(row):
    plot_type = row["plot_type"]
    expr = "value" if plot_type == "hist" else "value:x"
    adf = _make_eager_adf()
    kwargs = {
        "type": plot_type,
        "selection_vector": ["sel0", "sel1"],
        "normalize": "ratio",
        "auto_title": False,
    }
    if plot_type != "scatter":
        kwargs["bins"] = BINS

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = adf.draw(expr, **kwargs)
        except Exception as exc:
            # A repaired implementation must be a deterministic API refusal.
            # Return normally only for an acceptable refusal; strict XPASS then
            # forces the contract/current_state to be updated before banking.
            msg = str(exc).lower()
            if isinstance(exc, (ValueError, NotImplementedError)) and "normalize" in msg:
                return
            raise TypeError(
                f"unexpected non-profile normalize refusal for {plot_type}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    try:
        expected_warning = row["observed_warning_fragment"]
        assert any(expected_warning in str(w.message) for w in caught), (
            f"{plot_type}: current gap no longer matches warn+ignore signature: "
            f"{[str(w.message) for w in caught]}"
        )
        # Current behavior continued instead of refusing.  Verify normalize was
        # not accidentally applied, then emit the exact strict-XFAIL signature.
        stats = result[2]
        if isinstance(stats, dict):
            assert "normalize_data" not in stats
        else:
            assert not isinstance(stats, dict)
        raise AssertionError(CURRENT_NONPROFILE_ASSERT)
    finally:
        _close_result_figures(result)
