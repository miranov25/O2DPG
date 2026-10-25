"""PHASE_13_86_DF Stage B-b — expression-vector × ordinary facet composition.

These tests intentionally use asymmetric injected truth.  They prove the
semantic tuple

    facet × expression component × group × x-bin

against direct pandas/NumPy calculations.  They also characterize neighboring
hist/scatter and N-D facet combinations as loud refusals rather than allowing
an accidental generic-vector/same=True failure path.
"""
from __future__ import annotations

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from dfextensions.dfdraw.drawer import DFDraw


EXPR = "[y0,y1]:x"
XRANGE = (-0.5, 3.5)


def _truth_df() -> pd.DataFrame:
    rows = []
    for facet_i, facet in enumerate(("A", "B")):
        for group_i, group in enumerate(("g0", "g1")):
            for x in range(4):
                for rep in range(3):
                    rows.append(
                        {
                            "facet": facet,
                            "facet_num": 0.10 + 0.80 * facet_i,
                            "group": group,
                            "occ": 10.0 + 50.0 * group_i + 2.0 * x + rep,
                            "x": float(x),
                            # Distinct facet/group/x slopes and distinct
                            # within-bin spreads make branch/facet/group swaps
                            # visible in mean AND SEM.
                            "y0": float(10 * facet_i + 3 * group_i + 2 * x + rep),
                            "y1": float(100 + 20 * facet_i + 7 * group_i + 5 * x + 2 * rep),
                            "y2": float(250 + 30 * facet_i + 11 * group_i + 7 * x + 3 * rep),
                            "sel0": bool(x < 2),
                            "sel1": bool(x >= 2),
                            "w0": 1.0,
                            "w1": float(1 + rep),
                        }
                    )
    return pd.DataFrame(rows)


def _expected(df: pd.DataFrame, y: str, facet: str, group: str | None):
    sub = df[df["facet"] == facet]
    keys = ["x"] if group is None else ["group", "x"]
    out = {}
    for key, part in sub.groupby(keys, sort=True):
        if group is None:
            x = float(key if not isinstance(key, tuple) else key[0])
            g = None
        else:
            g, x = key
            if str(g) != group:
                continue
            x = float(x)
        arr = part[y].to_numpy(dtype=float)
        out[x] = {
            "count": int(arr.size),
            "mean": float(arr.mean()),
            "sem": float(arr.std(ddof=1) / math.sqrt(arr.size)),
        }
    return out


def _profile_rows(branch_stats: dict, facet: str, group: str | None):
    cell = branch_stats["per_group"][str(facet)]
    data = cell["profile_data"].copy()
    if group is not None:
        data = data[data["group"].astype(str) == str(group)]
    return data.sort_values("x_center").reset_index(drop=True)


def _assert_branch_oracle(stats, df, branch_i: int, y: str, *, grouped: bool):
    branch = stats[branch_i]
    assert branch["faceted"] is True
    assert branch["facet_by"] == "facet"
    assert list(branch["groups"]) == ["A", "B"]
    for facet in ("A", "B"):
        groups = ("g0", "g1") if grouped else (None,)
        for group in groups:
            expected = _expected(df, y, facet, group)
            actual = _profile_rows(branch, facet, group)
            assert len(actual) == 4
            for row in actual.itertuples(index=False):
                x = float(row.x_center)
                exp = expected[x]
                assert int(row.count) == exp["count"]
                assert float(row.y_mean) == pytest.approx(exp["mean"], abs=1e-12)
                assert float(row.y_sem) == pytest.approx(exp["sem"], abs=1e-12)


def test_BB1_typed_profile_vector_ordinary_facet_matches_independent_oracle():
    df = _truth_df()
    fig, axes, stats = DFDraw(df).profile(
        EXPR,
        bins=4,
        range=XRANGE,
        facet_by="facet",
        return_data=True,
        min_entries=1,
        error="sem",
    )
    try:
        assert isinstance(stats, list) and len(stats) == 2
        _assert_branch_oracle(stats, df, 0, "y0", grouped=False)
        _assert_branch_oracle(stats, df, 1, "y1", grouped=False)
        assert sum(ax.get_visible() for ax in np.asarray(axes).ravel()) == 2
    finally:
        plt.close(fig)


def test_BB2_draw_group_vector_ordinary_facet_matches_oracle_and_typed_owner():
    df = _truth_df()
    kwargs = dict(
        bins=4,
        range=XRANGE,
        group_by="group",
        facet_by="facet",
        return_data=True,
        min_entries=1,
        error="sem",
    )
    fig_t, axes_t, stats_t = DFDraw(df).profile(EXPR, **kwargs)
    fig_d, axes_d, stats_d = DFDraw(df).draw(EXPR, type="profile", **kwargs)
    try:
        for stats in (stats_t, stats_d):
            assert isinstance(stats, list) and len(stats) == 2
            _assert_branch_oracle(stats, df, 0, "y0", grouped=True)
            _assert_branch_oracle(stats, df, 1, "y1", grouped=True)
        # Public draw() must reach the same typed owner semantics, not a second
        # implementation.  Compare the machine-readable profile tables.
        for bi in range(2):
            for facet in ("A", "B"):
                pd.testing.assert_frame_equal(
                    stats_d[bi]["per_group"][facet]["profile_data"].reset_index(drop=True),
                    stats_t[bi]["per_group"][facet]["profile_data"].reset_index(drop=True),
                    check_exact=False,
                    rtol=1e-13,
                    atol=1e-13,
                )
    finally:
        plt.close(fig_t)
        plt.close(fig_d)


def test_BB3_realistic_group_quantiles_and_facet_bins_compose_inside_panels():
    df = _truth_df()
    # Mirror the production bug shape: three expression components, quantile
    # groups, and binned ordinary facets.
    fig, axes, stats = DFDraw(df).draw(
        "[y0,y1,y2]:x",
        type="profile",
        bins=4,
        range=XRANGE,
        group_by="occ",
        group_by_quantiles=3,
        facet_by="facet_num",
        facet_by_bins=2,
        return_data=True,
        min_entries=1,
        auto_title=True,
    )
    try:
        assert isinstance(stats, list) and len(stats) == 3
        for branch in stats:
            assert branch["n_groups"] == 2
            assert len(branch["groups"]) == 2
            for facet_key in branch["groups"]:
                data = branch["per_group"][str(facet_key)]["profile_data"]
                assert data["group"].nunique() == 3
        assert fig._suptitle is not None
    finally:
        plt.close(fig)


@pytest.mark.parametrize("compose,expected_branches", [("inner", 2), ("outer", 4)])
def test_BB4_expression_vector_selection_vector_composes_within_each_facet(
    compose, expected_branches
):
    df = _truth_df()
    fig, _, stats = DFDraw(df).profile(
        EXPR,
        bins=4,
        range=XRANGE,
        facet_by="facet",
        selection_vector=["sel0", "sel1"],
        vector_compose=compose,
        return_data=True,
        min_entries=1,
    )
    try:
        assert isinstance(stats, list) and len(stats) == expected_branches
        for branch in stats:
            assert branch["groups"] == ["A", "B"]
            # Each selection branch must actually filter rows: no branch may
            # retain all four x bins in every facet.
            n_populated = [
                int((branch["per_group"][facet]["profile_data"]["count"] > 0).sum())
                for facet in ("A", "B")
            ]
            assert n_populated == [2, 2]
    finally:
        plt.close(fig)


@pytest.mark.parametrize("compose,expected_branches", [("inner", 2), ("outer", 4)])
def test_BB5_expression_vector_weights_vector_composes_within_each_facet(
    compose, expected_branches
):
    df = _truth_df()
    fig, _, stats = DFDraw(df).profile(
        EXPR,
        bins=4,
        range=XRANGE,
        facet_by="facet",
        weights_vector=["w0", "w1"],
        vector_compose=compose,
        return_data=True,
        min_entries=1,
    )
    try:
        assert isinstance(stats, list) and len(stats) == expected_branches
        for branch in stats:
            assert branch["groups"] == ["A", "B"]
    finally:
        plt.close(fig)


@pytest.mark.parametrize("door", ["typed", "draw"])
@pytest.mark.parametrize("kind,expr", [("hist", "[y0,y1]"), ("scatter", EXPR)])
def test_BB6_neighbor_hist_scatter_expression_vector_facet_refuse_loudly(
    door, kind, expr
):
    df = _truth_df()
    d = DFDraw(df)
    with pytest.raises(NotImplementedError, match="expression-vector.*facet_by"):
        if door == "draw":
            d.draw(expr, type=kind, facet_by="facet")
        else:
            getattr(d, kind)(expr, facet_by="facet")


@pytest.mark.parametrize("door", ["typed", "draw"])
def test_BB7_profile_expression_vector_nd_facet_positive_contract(door):
    """PHASE_13_94: the historical Stage B-b refusal is now a positive proof."""
    df = _truth_df().assign(facet2=np.tile(["L", "R"], len(_truth_df()) // 2))
    d = DFDraw(df)
    opts = dict(facet_by=["facet", "facet2"], bins=4, range=XRANGE,
                min_entries=1, return_data=True)
    if door == "draw":
        fig, axes, stats = d.draw(EXPR, type="profile", **opts)
    else:
        fig, axes, stats = d.profile(EXPR, **opts)
    try:
        assert axes.shape == (2, 2)
        for facet in ("A", "B"):
            for facet2 in ("L", "R"):
                branches = stats[(facet, facet2)]
                assert isinstance(branches, list) and len(branches) == 2
                for branch, column in zip(branches, ("y0", "y1")):
                    tab = branch["profile_data"].sort_values("x_center")
                    truth = df[(df.facet == facet) & (df.facet2 == facet2)]
                    assert int(tab["count"].sum()) == len(truth)
                    assert np.allclose(tab["y_mean"].to_numpy(),
                                       truth.groupby("x")[column].mean().to_numpy(),
                                       rtol=0, atol=1e-12)
        assert fig is axes[0, 0].figure
    finally:
        plt.close(fig)


def test_BB8_public_same_true_neighbor_remains_invalid():
    df = _truth_df()
    with pytest.raises(ValueError, match="facet_by and same=True are mutually exclusive"):
        DFDraw(df).profile(EXPR, facet_by="facet", same=True)

def test_BB9_expression_vector_channel_identity_is_stable_across_facets():
    """Same expression component keeps the same visual channel in every facet."""
    df = _truth_df()
    fig, axes, _ = DFDraw(df).profile(
        EXPR,
        bins=4,
        range=XRANGE,
        facet_by="facet",
        min_entries=1,
        error="sem",
    )
    try:
        panel_colors = []
        for ax in np.asarray(axes, dtype=object).ravel():
            if not ax.get_visible():
                continue
            main_lines = [
                line for line in ax.lines
                if line.get_marker() == "o" and line.get_linestyle() == "-"
            ]
            assert len(main_lines) == 2
            colors = [tuple(line.get_color()) if not isinstance(line.get_color(), str)
                      else line.get_color() for line in main_lines]
            assert colors[0] != colors[1]
            panel_colors.append(colors)
        assert len(panel_colors) == 2
        assert panel_colors[0] == panel_colors[1]
    finally:
        plt.close(fig)
