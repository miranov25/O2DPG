"""PHASE_13_88_DF Bug B + F7 profile-normalize regressions."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from dfextensions.dfdraw import DFDraw
from dfextensions.dfdraw.plots._autorange import resolve_range_1d
from dfextensions.dfdraw.style import get_style_value


def _fixture():
    x = np.linspace(-1.0, 1.0, 1200)
    y = 12.0 + 3.0 * x + 0.2 * np.sin(np.arange(len(x)))
    return pd.DataFrame({
        "x": x,
        "y": y,
        "g": np.where(np.arange(len(x)) % 2 == 0, "A", "B"),
        "facet": np.where(x < 0, "L", "R"),
    })


def _half_plain(stats):
    nd = stats["normalize_data"]
    mask = ~nd["mask_undefined"].to_numpy(dtype=bool)
    values = nd.loc[mask, "value"].to_numpy(dtype=float)
    assert values.size > 0
    np.testing.assert_allclose(values, 0.5, rtol=1e-12, atol=1e-12)


def _half_nested(mapping):
    assert mapping
    for item in mapping.values():
        values = np.asarray(item["values"], dtype=float)
        mask = ~np.asarray(item["mask_undefined"], dtype=bool)
        assert np.count_nonzero(mask) > 0
        np.testing.assert_allclose(values[mask], 0.5, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("expr", ["[y,2*y]:x", "[y,2*y]:2*x"])
@pytest.mark.parametrize("owner", ["plain", "grouped", "faceted"])
def test_B1_profile_normalize_materializes_computed_positional_expressions(expr, owner):
    df = _fixture()
    kwargs = dict(
        bins=16,
        range=(-2.0, 2.0),
        normalize="ratio",
        return_data=True,
    )
    if owner == "grouped":
        kwargs["group_by"] = "g"
    elif owner == "faceted":
        kwargs["facet_by"] = "facet"

    fig, _, stats = DFDraw(df).profile(expr, **kwargs)
    try:
        if owner == "plain":
            _half_plain(stats)
        elif owner == "grouped":
            _half_nested(stats["normalize_data_grouped"])
        else:
            _half_nested(stats["normalize_data_faceted"])
    finally:
        plt.close(fig)


def test_B2_single_y_computed_x_supported_with_selection_vector():
    df = _fixture()
    fig, _, stats = DFDraw(df).profile(
        "y:2*x",
        selection_vector=["g=='A'", "g=='B'"],
        vector_compose="outer",
        bins=16,
        range=(-2.0, 2.0),
        normalize="ratio",
        return_data=True,
    )
    try:
        assert "normalize_data" in stats
    finally:
        plt.close(fig)


def _canonical_range(values):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return resolve_range_1d(
        None,
        finite,
        style_strategy=get_style_value("autorange.strategy", "hybrid"),
        style_k_robust=get_style_value("autorange.k_robust", 4.0),
        style_k_outlier=get_style_value("autorange.k_outlier", 1.5),
        style_percentile=get_style_value(
            "autorange.percentile", (1.0, 99.0)
        ),
    )[0]


def _assert_centers_use_range(centers, expected, bins):
    expected_edges = np.linspace(expected[0], expected[1], bins + 1)
    expected_centers = 0.5 * (expected_edges[:-1] + expected_edges[1:])
    np.testing.assert_allclose(
        np.asarray(centers, dtype=float),
        expected_centers,
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize("owner", ["plain", "grouped", "faceted"])
def test_F7_profile_normalize_uses_canonical_shared_autorange_not_raw_minmax(owner):
    """F7 owner-completeness: all three normalize owners use canonical range."""
    rng = np.random.default_rng(13887)
    x_core = rng.normal(0.0, 1.0, 2500)
    x = np.r_[x_core, 120.0]

    y = 10.0 + 0.1 * np.arange(len(x))
    idx = np.arange(len(x))
    df = pd.DataFrame({
        "x": x,
        "y": y,
        "y2": 2.0 * y,
        "g": np.where(idx % 2 == 0, "A", "B"),
        "facet": np.where(idx % 3 == 0, "L", "R"),
    })

    expected = _canonical_range(df["x"].to_numpy())
    raw_minmax = (float(np.min(x)), float(np.max(x)))
    assert not np.allclose(expected, raw_minmax)

    bins = 20
    kwargs = dict(
        bins=bins,
        normalize="ratio",
        return_data=True,
        # Deliberately no explicit range: this is the F7 contract under test.
    )
    if owner == "grouped":
        kwargs["group_by"] = "g"
    elif owner == "faceted":
        kwargs["facet_by"] = "facet"

    fig, _, stats = DFDraw(df).profile("[y,y2]:x", **kwargs)
    try:
        if owner == "plain":
            _assert_centers_use_range(
                stats["normalize_data"]["x_center"].to_numpy(dtype=float),
                expected,
                bins,
            )
        elif owner == "grouped":
            mapping = stats["normalize_data_grouped"]
            assert set(mapping) == {"A", "B"}
            for item in mapping.values():
                _assert_centers_use_range(item["bin_centers"], expected, bins)
        else:
            mapping = stats["normalize_data_faceted"]
            assert set(mapping) == {"L", "R"}
            for item in mapping.values():
                _assert_centers_use_range(item["bin_centers"], expected, bins)
    finally:
        plt.close(fig)
