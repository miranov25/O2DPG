"""PHASE_13_86_DF Stage B-c — computed expressions inside profile vectors.

Stage B-c closes the remaining assumption that every bracket-vector component
must already be a physical DataFrame column.  It covers both Stage B-a
``facet_by='vector'`` and Stage B-b ordinary-facet routing, through typed and
``draw()`` doors, and also locks a computed shared x expression.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from dfextensions.dfdraw import DFDraw


def _df() -> pd.DataFrame:
    rows = []
    for facet_i, facet in enumerate(("A", "B")):
        for x in range(4):
            for rep in range(3):
                rows.append({
                    "facet": facet,
                    "x": float(x),
                    "y0": float(10 * facet_i + 2 * x + rep),
                    "y1": float(100 + 20 * facet_i + 5 * x + 2 * rep),
                })
    return pd.DataFrame(rows)


def _main_line_y(ax):
    ys = []
    for line in ax.lines:
        if line.get_marker() == "o" and len(line.get_ydata()):
            ys.append(np.asarray(line.get_ydata(), dtype=float))
    assert ys
    return ys


@pytest.mark.parametrize("door", ["typed", "draw"])
def test_BC1_computed_y_component_works_with_vector_facets(door):
    df = _df()
    d = DFDraw(df)
    kwargs = dict(bins=4, range=(-0.5, 3.5), facet_by="vector", min_entries=1)
    if door == "typed":
        fig, axes, stats = d.profile("[y0,y1-5]:x", **kwargs)
    else:
        fig, axes, stats = d.draw("[y0,y1-5]:x", type="profile", **kwargs)
    try:
        axes = np.asarray(axes, dtype=object).ravel()
        assert [ax.get_title() for ax in axes] == ["vector=y0", "vector=y1-5"]
        assert stats["groups"] == ["y0", "y1-5"]
        y0 = np.concatenate(_main_line_y(axes[0]))
        y1m5 = np.concatenate(_main_line_y(axes[1]))
        # Independent expression oracle: the second panel must render y1-5,
        # not the raw y1 column and not a literal column lookup.
        assert np.nanmean(y1m5) > np.nanmean(y0)
        assert np.nanmean(y1m5) < df["y1"].mean()
    finally:
        plt.close(fig)


@pytest.mark.parametrize("door", ["typed", "draw"])
def test_BC2_computed_y_component_works_inside_ordinary_facets(door):
    df = _df()
    d = DFDraw(df)
    kwargs = dict(
        bins=4,
        range=(-0.5, 3.5),
        facet_by="facet",
        return_data=True,
        min_entries=1,
    )
    if door == "typed":
        fig, _, stats = d.profile("[y0,y1-5]:x", **kwargs)
    else:
        fig, _, stats = d.draw("[y0,y1-5]:x", type="profile", **kwargs)
    try:
        assert isinstance(stats, list) and len(stats) == 2
        expr_branch = stats[1]
        for facet in ("A", "B"):
            actual = expr_branch["per_group"][facet]["profile_data"].sort_values("x_center")
            sub = df[df["facet"] == facet].copy()
            expected = (
                sub.assign(expr=sub["y1"] - 5.0)
                .groupby("x", sort=True)["expr"]
                .mean()
                .to_numpy(dtype=float)
            )
            np.testing.assert_allclose(
                actual["y_mean"].to_numpy(dtype=float), expected, rtol=0, atol=1e-12
            )
    finally:
        plt.close(fig)


@pytest.mark.parametrize("facet_by", ["vector", "facet"])
@pytest.mark.parametrize("door", ["typed", "draw"])
def test_BC3_computed_shared_x_expression_is_materialized(facet_by, door):
    df = _df()
    d = DFDraw(df)
    kwargs = dict(
        bins=4,
        range=(-1.0, 7.0),
        facet_by=facet_by,
        min_entries=1,
        return_data=True,
    )
    if door == "typed":
        fig, _, stats = d.profile("[y0,y1-5]:x*2", **kwargs)
    else:
        fig, _, stats = d.draw("[y0,y1-5]:x*2", type="profile", **kwargs)
    try:
        if facet_by == "vector":
            assert stats["groups"] == ["y0", "y1-5"]
        else:
            assert isinstance(stats, list) and len(stats) == 2
            assert stats[0]["groups"] == ["A", "B"]
    finally:
        plt.close(fig)
