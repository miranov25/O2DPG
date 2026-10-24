"""PHASE_13_93_DF Stage A RED oracles for dense N-D facet presentation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.text import Text

from dfextensions.dfdraw import DFDraw

ROW_ORDER = ["Base", "Rate", "TPC", "Unfm", "Ft0"]
COL_ORDER = [
    "Constant", "Tgl", "Qpt", "TglQpt",
    "MdEdx", "TglMdEdx", "MdEdxQpt", "TglMdEdxQpt",
]


def _dense_df(rows=ROW_ORDER[:3], cols=COL_ORDER[:3], figs=None, n_per_cell=8):
    records = []
    fig_values = ["F0"] if figs is None else list(figs)
    k = 0
    for f_i, figv in enumerate(fig_values):
        for r_i, row in enumerate(rows):
            for c_i, col in enumerate(cols):
                for point in range(n_per_cell):
                    x = float(point)
                    y = 100.0 * r_i + 10.0 * c_i + 0.25 * point + 1000.0 * f_i
                    records.append((x, y, row, col, figv, k))
                    k += 1
    df = pd.DataFrame(
        records, columns=["x", "y", "row_dim", "col_dim", "fig_dim", "row_id"]
    )
    df["row_dim"] = pd.Categorical(df["row_dim"], categories=list(rows), ordered=True)
    df["col_dim"] = pd.Categorical(df["col_dim"], categories=list(cols), ordered=True)
    df["fig_dim"] = pd.Categorical(df["fig_dim"], categories=fig_values, ordered=True)
    return df


def _close_result(result):
    figs = result[0]
    if isinstance(figs, list):
        for fig in figs:
            plt.close(fig)
    else:
        plt.close(figs)


def _count_figure_tight_layout(monkeypatch, render):
    calls = []
    original = Figure.tight_layout

    def wrapped(self, *args, **kwargs):
        calls.append(id(self))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "tight_layout", wrapped)
    result = render()
    return calls, result


def _cell_keys(stats):
    return [k for k in stats if isinstance(k, tuple) and len(k) == 2]


def _ordered_unique(values):
    result = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def _figure_texts(fig):
    return [
        artist.get_text()
        for artist in fig.findobj(match=Text)
        if artist.get_visible() and artist.get_text()
    ]


def test_A_R1_2d_nd_facet_has_exactly_one_whole_figure_layout_pass(monkeypatch):
    df = _dense_df(rows=ROW_ORDER[:2], cols=COL_ORDER[:3])

    def render():
        return DFDraw(df).scatter(
            "y:x", facet_by=["row_dim", "col_dim"],
            share_x="none", share_y="none",
        )

    calls, result = _count_figure_tight_layout(monkeypatch, render)
    try:
        assert len(calls) == 1, (
            "2-D N-D facet Figure must have exactly one whole-Figure "
            f"tight_layout pass; observed {len(calls)}"
        )
    finally:
        _close_result(result)


def test_A_R2_3d_nd_facet_has_one_layout_pass_per_generated_figure(monkeypatch):
    df = _dense_df(
        rows=ROW_ORDER[:2], cols=COL_ORDER[:2],
        figs=["F0", "F1"], n_per_cell=6,
    )

    def render():
        return DFDraw(df).scatter(
            "y:x", facet_by=["row_dim", "col_dim", "fig_dim"],
            share_x="none", share_y="none", share_across_figures=False,
        )

    calls, result = _count_figure_tight_layout(monkeypatch, render)
    figures = result[0]
    try:
        assert len(figures) == 2
        assert len(calls) == len(figures), (
            "3-D facets must have one whole-Figure layout pass per generated "
            f"Figure; {len(figures)} Figures produced {len(calls)} passes"
        )
    finally:
        _close_result(result)


def test_A_R3_ordered_categorical_rows_preserve_declared_order():
    df = _dense_df(rows=ROW_ORDER, cols=COL_ORDER[:2], n_per_cell=3)
    fig, axes, stats = DFDraw(df).scatter(
        "y:x", facet_by=["row_dim", "col_dim"],
        share_x="none", share_y="none",
    )
    try:
        observed = _ordered_unique([key[0] for key in _cell_keys(stats)])
        assert observed == ROW_ORDER
    finally:
        plt.close(fig)


def test_A_R4_ordered_categorical_columns_preserve_declared_order():
    df = _dense_df(rows=ROW_ORDER[:2], cols=COL_ORDER, n_per_cell=3)
    fig, axes, stats = DFDraw(df).scatter(
        "y:x", facet_by=["row_dim", "col_dim"],
        share_x="none", share_y="none",
    )
    try:
        observed = _ordered_unique([key[1] for key in _cell_keys(stats)])
        assert observed == COL_ORDER
    finally:
        plt.close(fig)


def test_A_R5_explicit_facet_order_overrides_categorical_order():
    df = _dense_df(rows=ROW_ORDER[:3], cols=COL_ORDER[:3], n_per_cell=4)
    requested_rows = ["TPC", "Base", "Rate"]
    requested_cols = ["Qpt", "Constant", "Tgl"]
    fig, axes, stats = DFDraw(df).scatter(
        "y:x",
        facet_by=["row_dim", "col_dim"],
        facet_order={"row_dim": requested_rows, "col_dim": requested_cols},
        share_x="none", share_y="none",
    )
    try:
        keys = _cell_keys(stats)
        assert _ordered_unique([key[0] for key in keys]) == requested_rows
        assert _ordered_unique([key[1] for key in keys]) == requested_cols
    finally:
        plt.close(fig)


def test_A_R5b_invalid_facet_order_refuses_clearly():
    df = _dense_df(rows=ROW_ORDER[:2], cols=COL_ORDER[:2])
    with pytest.raises(ValueError, match=r"facet_order|Unknown|unknown|category|value"):
        DFDraw(df).scatter(
            "y:x",
            facet_by=["row_dim", "col_dim"],
            facet_order={"row_dim": ["Base", "NOT_A_REAL_LEVEL"]},
        )


def test_A_R6_outer_column_headings_appear_once_each():
    rows, cols = ROW_ORDER[:2], COL_ORDER[:3]
    df = _dense_df(rows=rows, cols=cols)
    fig, axes, stats = DFDraw(df).scatter(
        "y:x", facet_by=["row_dim", "col_dim"],
        facet_labels="outer", share_x="none", share_y="none",
    )
    try:
        texts = _figure_texts(fig)
        for value in cols:
            assert texts.count(value) == 1
    finally:
        plt.close(fig)


def test_A_R7_outer_row_headings_appear_once_each():
    rows, cols = ROW_ORDER[:3], COL_ORDER[:2]
    df = _dense_df(rows=rows, cols=cols)
    fig, axes, stats = DFDraw(df).scatter(
        "y:x", facet_by=["row_dim", "col_dim"],
        facet_labels="outer", share_x="none", share_y="none",
    )
    try:
        texts = _figure_texts(fig)
        for value in rows:
            assert texts.count(value) == 1
    finally:
        plt.close(fig)


def test_A_R8_outer_mode_removes_outer_dimension_text_from_panels():
    df = _dense_df(rows=ROW_ORDER[:2], cols=COL_ORDER[:2])
    fig, axes, stats = DFDraw(df).scatter(
        "y:x", facet_by=["row_dim", "col_dim"],
        facet_labels="outer", share_x="none", share_y="none",
    )
    try:
        for ax in axes.flat:
            panel_text = ax.get_title() + "\n" + ax.get_ylabel()
            assert "row_dim=" not in panel_text
            assert "col_dim=" not in panel_text
    finally:
        plt.close(fig)


def test_A_R9_shared_xlabel_is_one_figure_semantic_label():
    df = _dense_df(rows=ROW_ORDER[:2], cols=COL_ORDER[:3])
    fig, axes, stats = DFDraw(df).scatter(
        "y:x", facet_by=["row_dim", "col_dim"],
        shared_xlabel=True, share_x="none", share_y="none",
    )
    try:
        assert [ax.get_xlabel() for ax in axes.flat if ax.get_xlabel()] == []
        fig_texts = [t.get_text() for t in fig.texts if t.get_visible()]
        assert fig_texts.count("x") == 1
    finally:
        plt.close(fig)


def test_A_R10_shared_ylabel_preserves_independent_limits_and_ticks():
    df = _dense_df(rows=ROW_ORDER[:2], cols=COL_ORDER[:3])
    fig, axes, stats = DFDraw(df).scatter(
        "y:x", facet_by=["row_dim", "col_dim"],
        shared_ylabel=True, share_x="none", share_y="none",
    )
    try:
        assert [ax.get_ylabel() for ax in axes.flat if ax.get_ylabel()] == []
        fig_texts = [t.get_text() for t in fig.texts if t.get_visible()]
        assert fig_texts.count("y") == 1
        ylims = [tuple(np.round(ax.get_ylim(), 10)) for ax in axes.flat]
        assert len(set(ylims)) > 1
        for ax in axes.flat:
            assert any(
                tick.get_visible() and tick.get_text()
                for tick in ax.get_yticklabels()
            )
    finally:
        plt.close(fig)


def test_A_R11_explicit_figure_margins_survive_final_layout():
    df = _dense_df(rows=ROW_ORDER[:2], cols=COL_ORDER[:2])
    requested = [0.12, 0.91, 0.88, 0.14]  # left, right, top, bottom
    fig, axes, stats = DFDraw(df).scatter(
        "y:x", facet_by=["row_dim", "col_dim"],
        figure_margins=requested, share_x="none", share_y="none",
    )
    try:
        sp = fig.subplotpars
        np.testing.assert_allclose(
            [sp.left, sp.right, sp.top, sp.bottom],
            requested, rtol=0.0, atol=1e-12,
        )
    finally:
        plt.close(fig)


def test_A_R12_explicit_subplot_spacing_survives_final_layout():
    df = _dense_df(rows=ROW_ORDER[:2], cols=COL_ORDER[:2])
    requested = [0.37, 0.41]
    fig, axes, stats = DFDraw(df).scatter(
        "y:x", facet_by=["row_dim", "col_dim"],
        subplot_spacing=requested, share_x="none", share_y="none",
    )
    try:
        sp = fig.subplotpars
        np.testing.assert_allclose(
            [sp.wspace, sp.hspace], requested, rtol=0.0, atol=1e-12
        )
    finally:
        plt.close(fig)


def test_A_R13_legacy_2d_facet_presentation_unchanged_without_new_options():
    rows, cols = ["R0", "R1"], ["C0", "C1"]
    df = _dense_df(rows=rows, cols=cols)
    df["row_dim"] = df["row_dim"].astype(object)
    df["col_dim"] = df["col_dim"].astype(object)

    fig, axes, stats = DFDraw(df).scatter(
        "y:x", facet_by=["row_dim", "col_dim"],
        share_x="none", share_y="none",
    )
    try:
        assert axes.shape == (2, 2)
        assert axes[0, 0].get_title() == "col_dim=C0"
        assert axes[0, 1].get_title() == "col_dim=C1"
        assert axes[0, 0].get_ylabel().startswith("row_dim=R0")
        assert axes[1, 0].get_ylabel().startswith("row_dim=R1")
    finally:
        plt.close(fig)


@pytest.mark.parametrize("left_margin", [0.06, 0.08])
def test_A_R14_rendered_left_decorations_do_not_intersect(left_margin):
    """Renderer bbox oracle: shared ylabel < row headings < tick/axis text.

    The 0.06 margin is a falsifier for the old fixed x=.007/x=left-.04
    layout; a 5x8 dense grid is the real dEdxAI topology.
    """
    df = _dense_df(rows=ROW_ORDER, cols=COL_ORDER, n_per_cell=3)
    margins = [left_margin, .96, .89, .08]
    fig, axes, _ = DFDraw(df).scatter(
        'y:x', facet_by=['row_dim', 'col_dim'],
        share_x='none', share_y='none', facet_labels='outer',
        shared_xlabel='TPC region', shared_ylabel='Coefficient',
        figure_margins=margins, subplot_spacing=[.25, .30],
    )
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        pad = 1.5 * fig.dpi / 72.0
        assert [fig.subplotpars.left, fig.subplotpars.right,
                fig.subplotpars.top, fig.subplotpars.bottom] == pytest.approx(margins)
        ylabel = next(t for t in fig.texts if t.get_text() == 'Coefficient')
        ylabel_bbox = ylabel.get_window_extent(renderer)
        headings = [next(t for t in fig.texts if t.get_text() == r)
                    for r in ROW_ORDER]
        heading_bboxes = [t.get_window_extent(renderer) for t in headings]
        tight_left = min(ax.get_tightbbox(renderer).x0 for ax in axes[:, 0])
        assert ylabel_bbox.x0 >= 0
        assert all(ylabel_bbox.x1 + pad < box.x0 for box in heading_bboxes)
        assert all(box.x1 + pad < tight_left for box in heading_bboxes)
        assert fig._supxlabel.get_window_extent(renderer).y0 >= 0
    finally:
        plt.close(fig)


def test_A_R15_too_narrow_left_gutter_warns_without_changing_margins():
    """Explicit margins are authoritative even when decorations cannot fit."""
    df = _dense_df(rows=ROW_ORDER[:2], cols=COL_ORDER[:2])
    requested = [.025, .95, .89, .08]
    with pytest.warns(UserWarning, match='insufficient room|increase.*figure_margins'):
        fig, axes, _ = DFDraw(df).scatter(
            'y:x', facet_by=['row_dim', 'col_dim'], facet_labels='outer',
            shared_ylabel='Coefficient', shared_xlabel='Region',
            figure_margins=requested, share_x='none', share_y='none')
    try:
        sp = fig.subplotpars
        np.testing.assert_allclose(
            [sp.left, sp.right, sp.top, sp.bottom], requested, atol=1e-12,
            rtol=0)
    finally:
        plt.close(fig)
