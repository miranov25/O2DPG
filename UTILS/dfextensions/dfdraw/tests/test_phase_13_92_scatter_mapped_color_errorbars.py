"""PHASE_13_92_DF — mapped scatter color + error-bar composition.

Binding proof for the two-layer renderer:

    Axes.scatter()              owns point identity / mapped color / marker
    Axes.errorbar(fmt='none')   owns uncertainty geometry only

The tests intentionally inspect matplotlib artists and independent coordinates;
"no exception" is not accepted as a sufficient oracle.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib.container import ErrorbarContainer
import pytest

from dfextensions.dfdraw import DFDraw


def _df() -> pd.DataFrame:
    n = 8
    return pd.DataFrame({
        "x": np.arange(n, dtype=float),
        "y": 10.0 + np.arange(n, dtype=float),
        "c": np.linspace(-2.0, 2.0, n),
        "offset": np.linspace(0.1, 0.8, n),
        "ex": 0.05 + 0.01 * np.arange(n, dtype=float),
        "ey": 0.10 + 0.02 * np.arange(n, dtype=float),
        "facet": np.where(np.arange(n) < 4, "A", "B"),
        "mark": np.where(np.arange(n) % 2 == 0, "o", "s"),
        "category": np.where(np.arange(n) % 2 == 0, "left", "right"),
    })


def _point_collections(ax):
    return [c for c in ax.collections if isinstance(c, PathCollection)]


def _error_containers(ax):
    return [c for c in ax.containers if isinstance(c, ErrorbarContainer)]


def _bar_segments(container):
    return [np.asarray(c.get_segments(), dtype=float) for c in container.lines[2]]


def _close(fig):
    if isinstance(fig, (list, tuple)):
        for f in fig:
            plt.close(f)
    else:
        plt.close(fig)


def _assert_offsets(collection, x, y):
    got = np.asarray(collection.get_offsets(), dtype=float)
    expected = np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=0.0)


def _assert_yerr_segments(container, x, y, ey):
    segs = _bar_segments(container)
    assert len(segs) == 1, "yerr-only request must create one bar LineCollection"
    expected = np.asarray([
        [[xi, yi - ei], [xi, yi + ei]]
        for xi, yi, ei in zip(x, y, ey)
    ])
    np.testing.assert_allclose(segs[0], expected, rtol=0.0, atol=1e-12)


def _assert_xerr_segments(container, x, y, ex):
    segs = _bar_segments(container)
    assert len(segs) == 1, "xerr-only request must create one bar LineCollection"
    expected = np.asarray([
        [[xi - ei, yi], [xi + ei, yi]]
        for xi, yi, ei in zip(x, y, ex)
    ])
    np.testing.assert_allclose(segs[0], expected, rtol=0.0, atol=1e-12)


def test_R1_numeric_mapped_color_plus_yerr_two_layer():
    df = _df()
    fig, ax, _ = DFDraw(df).draw(
        "y:x", type="scatter", color="c", yerr="ey", colorbar=False
    )
    try:
        points = _point_collections(ax)
        errors = _error_containers(ax)
        assert len(points) == 1
        assert len(errors) == 1
        _assert_offsets(points[0], df.x, df.y)
        np.testing.assert_allclose(points[0].get_array(), df.c.to_numpy())
        _assert_yerr_segments(errors[0], df.x, df.y, df.ey)
        assert errors[0].lines[0] is None, "errorbar layer must not duplicate point markers"
    finally:
        _close(fig)


def test_R2_numeric_mapped_color_plus_xerr_two_layer():
    df = _df()
    fig, ax, _ = DFDraw(df).draw(
        "y:x", type="scatter", color="c", xerr="ex", colorbar=False
    )
    try:
        points = _point_collections(ax)
        errors = _error_containers(ax)
        assert len(points) == 1
        assert len(errors) == 1
        np.testing.assert_allclose(points[0].get_array(), df.c.to_numpy())
        _assert_xerr_segments(errors[0], df.x, df.y, df.ex)
        assert errors[0].lines[0] is None
    finally:
        _close(fig)


def test_R3_numeric_mapped_color_plus_both_errors_preserves_both_geometries():
    df = _df()
    fig, ax, _ = DFDraw(df).draw(
        "y:x", type="scatter", color="c", xerr="ex", yerr="ey", colorbar=False
    )
    try:
        points = _point_collections(ax)
        errors = _error_containers(ax)
        assert len(points) == 1
        assert len(errors) == 1
        _assert_offsets(points[0], df.x, df.y)
        np.testing.assert_allclose(points[0].get_array(), df.c.to_numpy())

        segs = _bar_segments(errors[0])
        assert len(segs) == 2, "xerr+yerr must create horizontal and vertical bar collections"
        expected_x = np.asarray([
            [[xi - ei, yi], [xi + ei, yi]]
            for xi, yi, ei in zip(df.x, df.y, df.ex)
        ])
        expected_y = np.asarray([
            [[xi, yi - ei], [xi, yi + ei]]
            for xi, yi, ei in zip(df.x, df.y, df.ey)
        ])
        np.testing.assert_allclose(segs[0], expected_x, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(segs[1], expected_y, rtol=0.0, atol=1e-12)
        assert errors[0].lines[0] is None
    finally:
        _close(fig)


@pytest.mark.parametrize("err_name", ["yerr", "xerr"])
def test_R4_R5_faceted_numeric_mapped_color_preserves_points_and_errors(err_name):
    df = _df()
    kwargs = {err_name: "ey" if err_name == "yerr" else "ex"}
    fig, axes, _ = DFDraw(df).draw(
        "y:x",
        type="scatter",
        color="c",
        facet_by="facet",
        ncols=2,
        colorbar=False,
        **kwargs,
    )
    try:
        flat = [a for a in np.asarray(axes, dtype=object).ravel() if a is not None]
        assert len(flat) >= 2
        for ax, facet in zip(flat[:2], ["A", "B"]):
            sub = df[df.facet == facet]
            points = _point_collections(ax)
            errors = _error_containers(ax)
            assert len(points) == 1
            assert len(errors) == 1
            _assert_offsets(points[0], sub.x, sub.y)
            np.testing.assert_allclose(points[0].get_array(), sub.c.to_numpy())
            if err_name == "yerr":
                _assert_yerr_segments(errors[0], sub.x, sub.y, sub.ey)
            else:
                _assert_xerr_segments(errors[0], sub.x, sub.y, sub.ex)
    finally:
        _close(fig)


def test_R6_computed_color_expression_plus_errorbar_and_colorbar():
    df = _df()
    expr = "2*c + offset"
    expected = df.eval(expr).to_numpy(dtype=float)
    fig, ax, _ = DFDraw(df).draw(
        "y:x", type="scatter", color=expr, yerr="ey", colorbar=True
    )
    try:
        points = _point_collections(ax)
        assert len(points) == 1
        np.testing.assert_allclose(points[0].get_array(), expected)
        assert points[0].colorbar is not None
        assert points[0].colorbar.mappable is points[0]
        assert points[0].colorbar.ax.get_ylabel() == expr
        assert len(_error_containers(ax)) == 1
    finally:
        _close(fig)


def test_R7_colorbar_mappable_is_exact_point_collection_object():
    df = _df()
    fig, ax, _ = DFDraw(df).draw(
        "y:x", type="scatter", color="c", yerr="ey", colorbar=True
    )
    try:
        points = _point_collections(ax)
        assert len(points) == 1
        cbar = points[0].colorbar
        assert cbar is not None
        assert cbar.mappable is points[0]
        assert cbar.ax is not ax
        assert _error_containers(ax)[0] is not cbar.mappable
    finally:
        _close(fig)


def test_R8_asymmetric_row_alignment_after_selection_nan_and_range_filter():
    df = pd.DataFrame({
        "x": [-3., -2., -1., 0., 1., 2., 3.],
        "y": [10., 11., np.nan, 13., 14., 15., 16.],
        "c": [100., 101., 102., 103., 104., 105., 106.],
        "ey": [0.1, 0.2, 9.9, 0.4, 0.5, 0.6, 0.7],
        "keep": [True, False, True, True, True, True, False],
    })
    fig, ax, stats = DFDraw(df).draw(
        "y:x",
        type="scatter",
        selection="keep",
        color="c",
        yerr="ey",
        range=((-0.5, 2.5), (12.0, 16.0)),
        colorbar=False,
    )
    try:
        expected_x = np.asarray([0., 1., 2.])
        expected_y = np.asarray([13., 14., 15.])
        expected_c = np.asarray([103., 104., 105.])
        expected_ey = np.asarray([0.4, 0.5, 0.6])

        points = _point_collections(ax)
        errors = _error_containers(ax)
        assert len(points) == 1
        assert len(errors) == 1
        _assert_offsets(points[0], expected_x, expected_y)
        np.testing.assert_allclose(points[0].get_array(), expected_c)
        _assert_yerr_segments(errors[0], expected_x, expected_y, expected_ey)
        assert stats["n"] == 3
    finally:
        _close(fig)


def test_R9_errorbars_have_explicitly_lower_zorder_than_points():
    df = _df()
    fig, ax, _ = DFDraw(df).draw(
        "y:x", type="scatter", color="c", yerr="ey", colorbar=False
    )
    try:
        point = _point_collections(ax)[0]
        err = _error_containers(ax)[0]
        bar_zorders = [bar.get_zorder() for bar in err.lines[2]]
        assert bar_zorders
        assert all(z < point.get_zorder() for z in bar_zorders)
    finally:
        _close(fig)


def test_R10_per_point_marker_expression_plus_yerr_keeps_marker_owner():
    df = _df()
    fig, ax, _ = DFDraw(df).draw(
        "y:x",
        type="scatter",
        color="tab:blue",
        marker="c > 0",
        yerr="ey",
        colorbar=False,
    )
    try:
        points = _point_collections(ax)
        errors = _error_containers(ax)
        assert len(points) == 2, "boolean marker expression must retain two point-marker collections"
        assert len(errors) == 1
        assert errors[0].lines[0] is None, "errorbar layer must be marker-free"
        assert sum(len(np.asarray(p.get_offsets())) for p in points) == len(df)
    finally:
        _close(fig)


def test_G1_string_categorical_color_plus_errors_remains_fail_loud():
    df = _df()
    with pytest.raises(ValueError, match="color mapping/expression with xerr/yerr error bars is unsupported"):
        DFDraw(df).draw(
            "y:x", type="scatter", color="category", yerr="ey", colorbar=False
        )


def test_C1_fixed_scalar_color_plus_errors_keeps_legacy_errorbar_owner():
    df = _df()
    fig, ax, _ = DFDraw(df).draw(
        "y:x", type="scatter", color="tab:red", xerr="ex", yerr="ey"
    )
    try:
        assert len(_error_containers(ax)) == 1
        assert len(_point_collections(ax)) == 0
        assert _error_containers(ax)[0].lines[0] is not None
    finally:
        _close(fig)


def test_C2_numeric_color_only_keeps_ordinary_scatter_path():
    df = _df()
    fig, ax, _ = DFDraw(df).draw(
        "y:x", type="scatter", color="c", colorbar=False
    )
    try:
        assert len(_point_collections(ax)) == 1
        assert len(_error_containers(ax)) == 0
        np.testing.assert_allclose(_point_collections(ax)[0].get_array(), df.c.to_numpy())
    finally:
        _close(fig)


def test_R11_caller_dataframe_not_mutated_by_composed_path():
    df = _df()
    before = df.copy(deep=True)
    fig, _, _ = DFDraw(df).draw(
        "y:x", type="scatter", color="2*c + offset", yerr="ey", colorbar=False
    )
    try:
        pd.testing.assert_frame_equal(df, before)
    finally:
        _close(fig)


def test_R12_marker_split_preserves_full_numeric_color_domain_and_colorbar_owner():
    """Real-data closure regression: marker split must not re-normalize color.

    This intentionally derives marker and color from the same binary variable.
    Without one shared Normalize, each marker subgroup contains a constant color
    and the last subgroup can collapse the colorbar to approximately 0.9..1.1
    instead of the full 0..1 domain.
    """
    df = _df().copy()
    df["side"] = (np.arange(len(df)) % 2).astype(float)

    fig, ax, _ = DFDraw(df).draw(
        "y:x",
        type="scatter",
        color="side",
        marker="side > 0.5",
        yerr="ey",
        colorbar=True,
    )
    try:
        points = _point_collections(ax)
        errors = _error_containers(ax)

        assert len(points) == 2, "binary marker expression must create two point collections"
        assert len(errors) == 1
        assert errors[0].lines[0] is None, "uncertainty layer must remain marker-free"

        # The two marker collections must share ONE normalization computed from
        # the full side={0,1} domain, not auto-scale subgroup-by-subgroup.
        assert points[0].norm is points[1].norm
        assert points[0].norm.vmin == pytest.approx(0.0)
        assert points[0].norm.vmax == pytest.approx(1.0)

        # Exactly one stable PathCollection owns the colorbar, and its norm is
        # the same shared full-domain norm used by every marker subgroup.
        colorbar_owners = [p for p in points if p.colorbar is not None]
        assert len(colorbar_owners) == 1
        owner = colorbar_owners[0]
        assert owner.colorbar.mappable is owner
        assert owner.norm is points[0].norm

        # Each subgroup still carries the correct original mapped values.
        got_values = sorted(
            tuple(np.unique(np.asarray(p.get_array(), dtype=float)))
            for p in points
        )
        assert got_values == [(0.0,), (1.0,)]
    finally:
        _close(fig)
