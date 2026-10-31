"""PHASE_13_101_ADF Slice B v0.1 — bounded synthetic RED/falsifier oracles."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from AliasDataFrame import AliasDataFrame
from dfextensions.dfdraw import set_style
from dfextensions.dfdraw.plots import _auto_title


@pytest.fixture(autouse=True)
def _close_figures():
    plt.close("all")
    yield
    plt.close("all")


def _paired_frame():
    rows = []
    for facet in ("A", "B", "C"):
        fshift = {"A": 0.0, "B": 2.0, "C": -1.0}[facet]
        for xb in range(4):
            for rep in range(6):
                rows.append(dict(
                    f=facet,
                    keep=(rep != 5),
                    x0=float(xb + rep * 0.08),
                    x1=float(50.0 + 10.0 * xb + rep * 0.7),
                    y0=float(fshift + 2.0 * xb + rep),
                    y1=float(100.0 + fshift + 3.0 * xb + 2.0 * rep),
                    y2=float(-20.0 + fshift + xb - rep),
                ))
    df = pd.DataFrame(rows)
    df.loc[(df["f"] == "B") & (df["x0"] < 1.0), "y0"] = np.nan
    df.loc[(df["f"] == "B") & (df["x1"] > 68.0), "y1"] = np.nan
    df.loc[(df["f"] == "C") & (df["x0"] > 2.0), "x0"] = np.inf
    return df


def _gvq_frame():
    rows = []
    for facet in range(6):
        for group in ("A", "B", "C"):
            for xb in range(6):
                for rep in range(8):
                    x = (xb + 0.10 + 0.08 * rep) / 6.0
                    rows.append(dict(
                        f=float(facet), g=group, x=float(x),
                        y0=float(0.4 * facet + 0.7 * xb
                                 + {"A": 0.0, "B": 1.5, "C": -1.0}[group]
                                 + 0.03 * rep),
                        y1=float(-0.2 * facet + 0.35 * xb
                                 + {"A": 2.0, "B": -0.5, "C": 1.0}[group]
                                 + 0.05 * rep),
                        keep=True,
                    ))
    return pd.DataFrame(rows)


def _raw_profile(df, x_name, y_name, edges, selection=None):
    work = df
    if selection is not None:
        work = work.loc[np.asarray(work.eval(selection), dtype=bool)]
    x = work[x_name].to_numpy(dtype=float)
    y = work[y_name].to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    rows = []
    for i in range(len(edges) - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        mask = (x >= lo) & ((x < hi) if i < len(edges) - 2 else (x <= hi))
        values = y[mask]
        rows.append(dict(
            x_low=lo, x_high=hi, x_center=(lo + hi) / 2.0,
            count=int(values.size),
            y_mean=float(np.mean(values)) if values.size else np.nan,
        ))
    return pd.DataFrame(rows)


def _assert_profile_core(observed, expected, atol=1e-12):
    got = observed.reset_index(drop=True)
    exp = expected.reset_index(drop=True)
    assert len(got) == len(exp)
    for col in ("x_low", "x_high", "x_center", "count", "y_mean"):
        np.testing.assert_allclose(
            got[col].to_numpy(), exp[col].to_numpy(),
            rtol=0, atol=atol, equal_nan=True, err_msg=col)


def _legend_labels(fig):
    return [t.get_text() for leg in fig.legends for t in leg.get_texts()]


def _assert_factored_gvq_labels(labels):
    required = ("y0", "y1", "A", "B", "C", "q=15%", "median", "q=90%")
    for item in required:
        assert labels.count(item) == 1, (item, labels)
    assert len(labels) == 8, labels


@pytest.mark.invariance
def test_A101_100_03_public_broadcast_contract_and_true_3_to_2_refusal():
    adf = AliasDataFrame(_paired_frame())
    fig, _, stats = adf.draw(
        "[y0,y1]:[x0]", type="profile", facet_by="f",
        bins=4, range=(0.0, 90.0), return_data=True, min_entries=1)
    assert isinstance(stats, list) and len(stats) == 2
    plt.close(fig)

    fig, _, stats = adf.draw(
        "y0:[x0,x1]", type="profile", facet_by="f",
        bins=4, range=(0.0, 90.0), return_data=True, min_entries=1)
    assert isinstance(stats, list) and len(stats) == 2
    plt.close(fig)

    before = tuple(plt.get_fignums())
    with pytest.raises(ValueError, match=r"broadcast|length|mismatch|N:N|N:1|1:N"):
        adf.draw(
            "[y0,y1,y2]:[x0,x1]", type="profile", facet_by="f",
            bins=4, range=(0.0, 90.0), return_data=True, min_entries=1)
    assert tuple(plt.get_fignums()) == before


@pytest.mark.invariance
def test_A101_100_02_public_paired_profile_uses_branch_local_finite_masks():
    df = _paired_frame()
    adf = AliasDataFrame(df.copy())
    fig, _, stats = adf.draw(
        "[y0,y1]:[x0,x1]", type="profile", facet_by="f",
        selection="keep", bins=4, range=(0.0, 90.0),
        return_data=True, min_entries=1)
    try:
        assert isinstance(stats, list) and len(stats) == 2
        edges = np.linspace(0.0, 90.0, 5)
        selected = df.loc[df["keep"]].copy()
        for branch_i, (y_name, x_name) in enumerate((("y0", "x0"), ("y1", "x1"))):
            branch = stats[branch_i]
            assert branch["faceted"] is True
            assert branch["facet_by"] == "f"
            for facet in branch["groups"]:
                observed = branch["per_group"][facet]["profile_data"]
                raw = selected.loc[selected["f"] == facet]
                expected = _raw_profile(raw, x_name, y_name, edges)
                _assert_profile_core(observed, expected)
        assert stats[0]["per_group"]["B"]["n"] != stats[1]["per_group"]["B"]["n"]
    finally:
        plt.close(fig)


@pytest.mark.invariance
def test_A101_99_02_companion_1d_global_group_vector_quantiles_are_raw_truthful():
    df = _gvq_frame()
    adf = AliasDataFrame(df.copy())
    fig, axes, _ = adf.draw(
        "[y0,y1]:x", type="profile", selection="keep",
        bins=6, range=(0.0, 1.0),
        facet_by="f", facet_by_bins=6, ncols=3,
        group_by="g", group_by_scope="global",
        quantiles=[0.15, 0.50, 0.90], quantile_mode="discrete",
        legend="shared", auto_title=False, min_entries=1, return_data=True)
    try:
        assert len(np.asarray(axes).flat) == 6
        assert len(fig.legends) == 1
        _assert_factored_gvq_labels(_legend_labels(fig))
        panel = np.asarray(axes).flat[0]
        subset = df[(df["f"] == 0.0) & (df["g"] == "A") & df["keep"]]
        edges = np.linspace(0.0, 1.0, 7)
        centers = (edges[:-1] + edges[1:]) / 2.0
        for expr in ("y0", "y1"):
            for q in (0.15, 0.50, 0.90):
                line = next(
                    ln for ln in panel.lines
                    if getattr(ln, "_dfdraw_facet_semantic", {}).get("kind") == "quantile"
                    and ln._dfdraw_facet_semantic["expression"] == expr
                    and ln._dfdraw_facet_semantic["group"] == "A"
                    and ln._dfdraw_facet_semantic["quantile"] == q)
                expected = []
                for i in range(6):
                    mask = ((subset["x"] >= edges[i]) &
                            ((subset["x"] < edges[i + 1]) if i < 5
                             else (subset["x"] <= edges[i + 1])))
                    values = subset.loc[mask, expr].to_numpy(dtype=float)
                    expected.append(
                        np.nanpercentile(values, q * 100.0, method="linear")
                        if values.size else np.nan)
                np.testing.assert_allclose(line.get_xdata(), centers, rtol=0, atol=1e-12)
                np.testing.assert_allclose(
                    line.get_ydata(), expected, rtol=0, atol=1e-12, equal_nan=True)
    finally:
        plt.close(fig)


@pytest.mark.invariance
def test_A101_98_01_public_figure_owns_auto_title_fit(monkeypatch):
    adf = AliasDataFrame(_gvq_frame())
    measured = []
    monkeypatch.setattr(
        _auto_title, "fit_auto_title_text",
        lambda text, *args, **kwargs: measured.append(str(text)) or text)
    fig, axes, _ = adf.draw(
        "[y0,y1]:x", type="profile", selection="keep",
        facet_by=["f", "x"], facet_by_bins=[3, 2],
        group_by="g", group_by_scope="global",
        bins=4, range=(0.0, 1.0),
        legend="shared", auto_title=True, min_entries=1)
    try:
        assert np.asarray(axes).shape == (3, 2)
        assert fig._suptitle is not None
        assert len(measured) == 1, measured
        assert all(not ax.get_title() for ax in np.asarray(axes).flat)
    finally:
        plt.close(fig)


@pytest.mark.invariance
def test_A101_97_02_explicit_underprecision_refuses_before_figure():
    rows = []
    for rowv in (0.0005, 0.0015, 0.0025):
        for colv in (-0.5, 0.5):
            for x in (0.2, 0.5, 0.8):
                for rep in range(4):
                    rows.append(dict(
                        rowv=rowv + (rep - 1.5) * 1e-5,
                        colv=colv + (rep - 1.5) * 0.002,
                        x=x, y=10.0 * rowv + colv + x + rep * 0.01))
    adf = AliasDataFrame(pd.DataFrame(rows))
    before = tuple(plt.get_fignums())
    with pytest.raises(ValueError, match=r"collision|distinct|precision"):
        adf.draw(
            "y:x", type="profile",
            facet_by=["rowv", "colv"], facet_by_bins=[3, 2],
            facet_labels="outer", facet_label_format={"rowv": ".2f"},
            bins=3, min_entries=1)
    assert tuple(plt.get_fignums()) == before


@pytest.mark.invariance
def test_A101_falsifier_wrong_x_substitution_is_detected():
    df = _paired_frame().loc[lambda x: x["keep"]].copy()
    edges = np.linspace(0.0, 90.0, 5)
    truth = _raw_profile(df, "x1", "y1", edges)
    wrong = _raw_profile(df, "x0", "y1", edges)
    with pytest.raises(AssertionError):
        _assert_profile_core(wrong, truth)


@pytest.mark.invariance
def test_A101_falsifier_missing_quantile_identity_is_detected():
    healthy = ["y0", "y1", "A", "B", "C", "q=15%", "median", "q=90%"]
    _assert_factored_gvq_labels(healthy)
    mutated = [x for x in healthy if x != "q=90%"]
    with pytest.raises(AssertionError):
        _assert_factored_gvq_labels(mutated)

# ---------------------------------------------------------------------------
# Slice-B v0.2 completion: remaining synthetic obligations before TS_ROOT
# ---------------------------------------------------------------------------

def _format_frame():
    rows = []
    for rowv in (100.0, 200.0, 300.0):
        for colv in (-0.75, -0.25, 0.25, 0.75):
            for xb in range(4):
                for rep in range(4):
                    rows.append(dict(
                        rowv=rowv + (rep - 1.5) * 0.1,
                        colv=colv + (rep - 1.5) * 0.002,
                        x=(xb + 0.15 + 0.05 * rep) / 4.0,
                        y=0.02 * rowv + 0.5 * colv + xb + 0.03 * rep,
                    ))
    return pd.DataFrame(rows)


def _sparse_gvq_frame():
    df = _gvq_frame()
    df = df[df["g"].isin(["A", "B"])].copy()
    # Independent second physical facet dimension. Every ordinary cell is
    # populated before the deliberate sparse-group deletion below.
    df["h"] = (df.index.to_numpy() % 2).astype(float)
    # Remove B from one physical cell only. Global identity must remain B,
    # never be relabelled or recoloured in the other cells.
    df = df[~((df["f"] < 2.0) & (df["h"] < 0.5) & (df["g"] == "B"))].copy()
    return df.reset_index(drop=True)


def _interval_headings(fig):
    # Interval labels never contain quotes; vector-expression Figure titles do.
    # Filter those titles out instead of counting them as facet headings.
    return [
        t.get_text()
        for t in fig.texts
        if t.get_text().startswith(("(", "["))
        and "'" not in t.get_text()
        and '"' not in t.get_text()
    ]


def _assert_nested_equal(a, b):
    assert type(a) is type(b), (type(a), type(b))
    if isinstance(a, pd.DataFrame):
        pd.testing.assert_frame_equal(
            a.reset_index(drop=True), b.reset_index(drop=True),
            check_dtype=True, check_exact=True)
    elif isinstance(a, pd.Series):
        pd.testing.assert_series_equal(
            a.reset_index(drop=True), b.reset_index(drop=True),
            check_dtype=True, check_exact=True)
    elif isinstance(a, dict):
        assert list(a.keys()) == list(b.keys())
        for key in a:
            _assert_nested_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _assert_nested_equal(x, y)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, float) and np.isnan(a):
        assert isinstance(b, float) and np.isnan(b)
    else:
        assert a == b


@pytest.mark.invariance
def test_A101_97_01_omitted_none_and_style_formatting_preserve_numeric_state():
    """C08 freeze: omitted/default and explicit None are legacy; style may override omission."""
    adf = AliasDataFrame(_format_frame())
    kwargs = dict(
        type="profile",
        facet_by=["rowv", "colv"],
        facet_by_bins=[3, 2],
        facet_labels="outer",
        bins=4,
        range=(0.0, 1.0),
        min_entries=1,
        return_data=True,
        legend="shared",
    )
    set_style(None)
    try:
        fig0, _, st0 = adf.draw("y:x", **kwargs)
        headings0 = _interval_headings(fig0)

        fig_none, _, st_none = adf.draw(
            "y:x", facet_label_format=None, **kwargs)
        headings_none = _interval_headings(fig_none)

        set_style({"facet.label_format": ".2f"})
        fig_style, _, st_style = adf.draw("y:x", **kwargs)
        headings_style = _interval_headings(fig_style)

        assert headings0 == headings_none
        assert headings_style != headings0
        assert len(headings_style) == len(headings0) == 5
        _assert_nested_equal(st0, st_none)
        _assert_nested_equal(st0, st_style)
    finally:
        set_style(None)
        plt.close("all")


@pytest.mark.invariance
def test_A101_98_02_sparse_global_group_identity_and_explicit_title_are_stable():
    """Sparse physical cells must not rewrite Figure-global group identity."""
    df = _sparse_gvq_frame()
    adf = AliasDataFrame(df)
    fig, axes, _ = adf.draw(
        "[y0,y1]:x",
        type="profile",
        facet_by=["f", "h"],
        facet_by_bins=[3, 2],
        group_by="g",
        group_by_scope="global",
        bins=6,
        range=(0.0, 1.0),
        legend="shared",
        title="Calibration QA",
        auto_title=False,
        min_entries=1,
    )
    try:
        assert np.asarray(axes).shape == (3, 2)
        assert fig._suptitle is not None
        assert fig._suptitle.get_text() == "Calibration QA"

        labels = _legend_labels(fig)
        assert labels.count("A") == 1
        assert labels.count("B") == 1
        assert all("__dfdraw_group_scope__" not in value for value in labels)

        colors = {"A": set(), "B": set()}
        presence = []
        for ax in np.asarray(axes).flat:
            groups_here = set()
            for line in ax.lines:
                meta = getattr(line, "_dfdraw_facet_semantic", {})
                group = meta.get("group")
                if group in colors:
                    groups_here.add(group)
                    colors[group].add(line.get_color())
            presence.append(groups_here)

        assert colors["A"] and colors["B"]
        assert len(colors["A"]) == 1
        assert len(colors["B"]) == 1
        assert any("A" in groups and "B" not in groups for groups in presence)
        assert any("A" in groups and "B" in groups for groups in presence)
    finally:
        plt.close(fig)


@pytest.mark.invariance
def test_A101_99_02_sparse_2d_group_vector_quantile_legend_is_factored():
    """Main 2-D sparse synthetic leg: E + G + Q, never Cartesian E×G×Q."""
    df = _sparse_gvq_frame()
    adf = AliasDataFrame(df)
    fig, axes, _ = adf.draw(
        "[y0,y1]:x",
        type="profile",
        facet_by=["f", "h"],
        facet_by_bins=[3, 2],
        facet_labels="outer",
        group_by="g",
        group_by_scope="global",
        bins=6,
        range=(0.0, 1.0),
        quantiles=[0.15, 0.50, 0.90],
        quantile_mode="discrete",
        legend="shared",
        auto_title=False,
        min_entries=1,
    )
    try:
        assert np.asarray(axes).shape == (3, 2)
        labels = _legend_labels(fig)
        required = ("y0", "y1", "A", "B", "q=15%", "median", "q=90%")
        for item in required:
            assert labels.count(item) == 1, (item, labels)
        assert len(labels) == 7, labels

        headings = _interval_headings(fig)
        assert len(headings) == 5  # 3 row + 2 column physical headings
        assert len(set(headings)) == 5
    finally:
        plt.close(fig)


@pytest.mark.invariance
@pytest.mark.parametrize("figsize", [(16, 9), (18, 11)])
def test_A101_99_03_figsize_and_explicit_shared_legend_anchor_are_preserved(figsize):
    """Renderer geometry synthetic leg for both mandatory Figure sizes."""
    adf = AliasDataFrame(_gvq_frame())
    fig, axes, _ = adf.draw(
        "[y0,y1]:x",
        type="profile",
        facet_by="f",
        facet_by_bins=6,
        ncols=3,
        bins=6,
        range=(0.0, 1.0),
        quantiles=[0.15, 0.50, 0.90],
        quantile_mode="discrete",
        figsize=figsize,
        legend={
            "mode": "shared",
            "loc": "lower left",
            "bbox_to_anchor": (0.30, 0.80),
            "ncol": 3,
            "title": "Explicit anchor test",
        },
        auto_title=True,
        min_entries=1,
    )
    try:
        np.testing.assert_allclose(fig.get_size_inches(), figsize, rtol=0, atol=1e-12)
        assert len(np.asarray(axes).flat) == 6
        assert len(fig.legends) == 1
        legend = fig.legends[0]
        assert legend.get_title().get_text() == "Explicit anchor test"
        assert legend._ncols == 3
        assert legend._loc == 3  # Matplotlib lower left

        anchor = legend.get_bbox_to_anchor().transformed(fig.transFigure.inverted())
        np.testing.assert_allclose([anchor.x0, anchor.y0], [0.30, 0.80],
                                   rtol=0, atol=1e-12)

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        lb = legend.get_window_extent(renderer)
        fb = fig.bbox
        assert fb.x0 <= lb.x0 <= lb.x1 <= fb.x1
        assert fb.y0 <= lb.y0 <= lb.y1 <= fb.y1
    finally:
        plt.close(fig)


@pytest.mark.invariance
def test_A101_100_01_computed_paired_x_minmax_edges_match_independent_raw_truth():
    """Independent Figure-domain minmax edges for two distinct positional X branches."""
    df = _paired_frame().copy()
    # Keep this leg focused on paired-X Figure-domain edges; asymmetric invalidity
    # is already locked by ADF100-02 above.
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["x0", "x1", "y0", "y1"]).copy()
    df["x1_shift"] = df["x1"] + 0.25
    selected = df.loc[df["keep"]].copy()

    adf = AliasDataFrame(df.drop(columns=["x1_shift"]).copy())
    fig, _, stats = adf.draw(
        "[y0,y1]:[x0,x1+0.25]",
        type="profile",
        selection="keep",
        facet_by="f",
        bins=4,
        range="minmax",
        quantiles=[0.15, 0.50, 0.90],
        quantile_mode="discrete",
        legend="shared",
        min_entries=1,
        return_data=True,
    )
    try:
        assert isinstance(stats, list) and len(stats) == 2

        specs = (
            ("y0", "x0"),
            ("y1", "x1_shift"),
        )
        branch_edges = []
        for branch_i, (y_name, x_name) in enumerate(specs):
            finite = (
                np.isfinite(selected[x_name].to_numpy(dtype=float))
                & np.isfinite(selected[y_name].to_numpy(dtype=float))
            )
            x_domain = selected.loc[finite, x_name].to_numpy(dtype=float)
            edges = np.linspace(float(x_domain.min()), float(x_domain.max()), 5)
            branch_edges.append(edges)

            branch = stats[branch_i]
            for facet in branch["groups"]:
                observed = branch["per_group"][facet]["profile_data"]
                raw = selected.loc[selected["f"] == facet]
                expected = _raw_profile(raw, x_name, y_name, edges)
                _assert_profile_core(observed, expected)

                got_edges = np.r_[
                    observed["x_low"].to_numpy(dtype=float),
                    observed["x_high"].to_numpy(dtype=float)[-1],
                ]
                np.testing.assert_allclose(got_edges, edges, rtol=0, atol=1e-12)

        assert not np.allclose(branch_edges[0], branch_edges[1], rtol=0, atol=1e-12)
    finally:
        plt.close(fig)
