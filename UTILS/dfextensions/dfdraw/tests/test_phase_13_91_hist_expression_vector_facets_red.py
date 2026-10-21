"""PHASE_13_91_DF — histogram expression-vector × facet_by="vector".

RED-first contract for the bounded additive histogram-vector-facet feature.

Important review constraints:
- hist() has no public return_data= contract; numerical truth is read from the
  rendered histogram artists and compared to independent numpy.histogram.
- computed expressions use the existing evaluator; no new parser.
- duplicate expression identities refuse before figure allocation.
- selection_vector / weights_vector cross-products and normalize= remain
  unsupported in this bounded phase unless explicitly ratified later.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from dfdraw.drawer import DFDraw


BINS = 8
RANGE = (-8.0, 14.0)
EXPRESSIONS = ("x-y", "2*x+y")


def _df():
    i = np.arange(80)
    x = np.linspace(-2.5, 3.5, len(i))
    y = 0.35 * x + (i % 5) * 0.4 - 0.6
    keep = (i % 3) != 0

    x = x.astype(float)
    y = y.astype(float)
    x[5] = np.nan
    y[7] = np.inf

    return pd.DataFrame({"x": x, "y": y, "keep": keep})


def _call(door, data, expr, **kwargs):
    d = DFDraw(data)
    if door == "typed":
        return d.hist(expr, **kwargs)
    if door == "draw":
        return d.draw(expr, type="hist", **kwargs)
    raise AssertionError(door)


def _visible_axes(axes):
    return [ax for ax in np.asarray(axes, dtype=object).ravel()
            if ax is not None and ax.get_visible()]


def _polygon_heights_for_edges(ax, edges):
    """Recover histogram heights from rendered bar/step artists."""
    edges = np.asarray(edges, dtype=float)
    mids = 0.5 * (edges[:-1] + edges[1:])
    heights = np.full(len(mids), np.nan, dtype=float)

    rects = [p for p in ax.patches
             if hasattr(p, "get_height")
             and hasattr(p, "get_x")
             and hasattr(p, "get_width")]
    usable_rects = [
        p for p in rects
        if np.isfinite(float(p.get_width())) and float(p.get_width()) > 0
    ]
    if len(usable_rects) >= len(mids):
        for j, mid in enumerate(mids):
            candidates = [
                float(p.get_height())
                for p in usable_rects
                if float(p.get_x()) - 1e-12 <= mid
                <= float(p.get_x()) + float(p.get_width()) + 1e-12
            ]
            if candidates:
                heights[j] = max(candidates)
        if np.all(np.isfinite(heights)):
            return heights

    for patch in ax.patches:
        if not hasattr(patch, "get_path"):
            continue
        verts = np.asarray(patch.get_path().vertices, dtype=float)
        if verts.ndim != 2 or verts.shape[0] < 2:
            continue
        for j, mid in enumerate(mids):
            candidates = []
            for p0, p1 in zip(verts[:-1], verts[1:]):
                x0, y0 = p0
                x1, y1 = p1
                if not np.isfinite([x0, y0, x1, y1]).all():
                    continue
                if not np.isclose(y0, y1, rtol=0.0, atol=1e-12):
                    continue
                lo, hi = sorted((x0, x1))
                if lo - 1e-12 <= mid <= hi + 1e-12:
                    candidates.append(float(y0))
            if candidates:
                h = max(candidates)
                heights[j] = h if not np.isfinite(heights[j]) else max(heights[j], h)

    assert np.all(np.isfinite(heights)), (
        "Could not recover all rendered histogram-bin heights from Matplotlib "
        f"artists: {heights}"
    )
    return heights


def _expected(df, expr):
    selected = df.loc[df["keep"]].copy()
    values = np.asarray(selected.eval(expr), dtype=float)
    values = values[np.isfinite(values)]
    return np.histogram(values, bins=BINS, range=RANGE)


@pytest.mark.parametrize("door", ["typed", "draw"])
def test_R1_R2_computed_expression_vector_facets_match_numpy_truth(door):
    """Both public doors: one panel per expression, exact independent counts."""
    df = _df()
    before_columns = tuple(df.columns)
    before_values = df.copy(deep=True)
    before_figs = set(plt.get_fignums())

    fig, axes, stats = _call(
        door,
        df,
        "[x-y,2*x+y]",
        selection="keep",
        facet_by="vector",
        bins=BINS,
        range=RANGE,
        ncols=2,
        auto_title=False,
    )
    try:
        data_axes = _visible_axes(axes)
        assert len(data_axes) == 2
        assert fig.number in (set(plt.get_fignums()) - before_figs)

        for ax, expr in zip(data_axes, EXPRESSIONS):
            expected_counts, expected_edges = _expected(df, expr)
            observed = _polygon_heights_for_edges(ax, expected_edges)
            np.testing.assert_allclose(
                observed, expected_counts.astype(float), rtol=0.0, atol=0.0
            )
            assert expr in ax.get_title()

        assert stats is not None
        pd.testing.assert_frame_equal(df, before_values, check_exact=True)
        assert tuple(df.columns) == before_columns
    finally:
        plt.close(fig)


def test_R3_typed_and_draw_doors_have_identical_panel_values():
    df = _df()
    kwargs = dict(
        selection="keep",
        facet_by="vector",
        bins=BINS,
        range=RANGE,
        ncols=2,
        auto_title=False,
    )
    fig_t, axes_t, stats_t = DFDraw(df).hist("[x-y,2*x+y]", **kwargs)
    fig_d, axes_d, stats_d = DFDraw(df).draw(
        "[x-y,2*x+y]", type="hist", **kwargs
    )
    try:
        at = _visible_axes(axes_t)
        ad = _visible_axes(axes_d)
        assert len(at) == len(ad) == 2
        for ax_t, ax_d, expr in zip(at, ad, EXPRESSIONS):
            _, edges = _expected(df, expr)
            np.testing.assert_allclose(
                _polygon_heights_for_edges(ax_t, edges),
                _polygon_heights_for_edges(ax_d, edges),
                rtol=0.0,
                atol=0.0,
            )
        assert type(stats_t) is type(stats_d)
    finally:
        plt.close(fig_t)
        plt.close(fig_d)


class _AxisTitleSource:
    def __init__(self):
        self.df = pd.DataFrame({
            "x": np.linspace(-2.0, 2.0, 60),
            "y": np.linspace(10.0, 20.0, 60),
        })
        self._titles = {
            "x": r"$X_{\rm metadata}$",
            "y": r"$Y_{\rm metadata}$",
        }

    def get_axis_title(self, name):
        return self._titles.get(name)


@pytest.mark.parametrize("door", ["typed", "draw"])
def test_R4_physical_column_vector_facets_preserve_component_x_metadata(door):
    """Binding review correction 91-C2: hist metadata belongs on x-axis."""
    src = _AxisTitleSource()
    fig, axes, _ = _call(
        door,
        src,
        "[x,y]",
        facet_by="vector",
        bins=6,
        range=(-5.0, 25.0),
        ncols=2,
        auto_title=False,
    )
    try:
        data_axes = _visible_axes(axes)
        assert len(data_axes) == 2
        assert [ax.get_xlabel() for ax in data_axes] == [
            r"$X_{\rm metadata}$",
            r"$Y_{\rm metadata}$",
        ]
    finally:
        plt.close(fig)


@pytest.mark.parametrize("door", ["typed", "draw"])
def test_R5_duplicate_expression_identity_refuses_before_figure_allocation(door):
    df = _df()
    before = set(plt.get_fignums())
    with pytest.raises((ValueError, NotImplementedError), match="duplicate|unique|expression"):
        _call(
            door,
            df,
            "[x-y,x-y]",
            selection="keep",
            facet_by="vector",
            bins=BINS,
            range=RANGE,
        )
    assert set(plt.get_fignums()) == before


@pytest.mark.parametrize(
    "extra, match",
    [
        ({"selection_vector": ["x<0", "x>=0"], "vector_compose": "outer"},
         "selection_vector|vector"),
        ({"weights_vector": ["x*0+1", "x*0+2"], "vector_compose": "outer"},
         "weights_vector|vector"),
        ({"normalize": "ratio"}, "normalize|facet"),
    ],
    ids=["selection-vector", "weights-vector", "normalize"],
)
def test_R6_deferred_cross_products_refuse_before_figure_allocation(extra, match):
    df = _df()
    before = set(plt.get_fignums())
    with pytest.raises((ValueError, NotImplementedError), match=match):
        DFDraw(df).hist(
            "[x-y,2*x+y]",
            facet_by="vector",
            bins=BINS,
            range=RANGE,
            **extra,
        )
    assert set(plt.get_fignums()) == before


def test_R7_nonfaceted_hist_expression_vector_overlay_remains_supported():
    """Existing vector overlay semantics must not be redirected to facets."""
    df = _df()
    fig, axes, stats = DFDraw(df).draw(
        "[x-y,2*x+y]",
        type="hist",
        selection="keep",
        bins=BINS,
        range=RANGE,
    )
    try:
        assert fig is not None
        assert stats is not None
        assert len(stats) == 2
    finally:
        plt.close(fig)


def test_R8_profile_expression_vector_facets_remain_supported():
    """Neighboring PHASE_13_86 owner must remain unchanged."""
    df = _df()
    fig, axes, stats = DFDraw(df).profile(
        "[x-y,2*x+y]:x",
        selection="keep",
        facet_by="vector",
        bins=6,
        range=(-2.5, 3.5),
        min_entries=1,
    )
    try:
        assert len(_visible_axes(axes)) == 2
        assert stats is not None
    finally:
        plt.close(fig)
