"""PHASE_13_86_DF Stage B-a — draw-door vector-facet delegation.

B-a is intentionally narrow: top-level ``draw()`` must preserve a bracket
profile expression until the existing typed ``profile()`` owner sees
``facet_by="vector"``.  Ordinary column/categorical facet composition is B-b.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dfextensions.dfdraw import DFDraw


def _asymmetric_vector_df():
    """Make y0/y1 and groups unmistakable if channels are swapped."""
    x = np.tile(np.arange(8, dtype=float), 4)
    g = np.repeat(["A", "B"], 16)
    y0 = 2.0 * x + np.where(g == "B", 50.0, 0.0)
    y1 = 100.0 + 5.0 * x + np.where(g == "B", 200.0, 0.0)
    return pd.DataFrame({"x": x, "y0": y0, "y1": y1, "g": g})


def _axis_line_payload(axes):
    payload = []
    for ax in np.asarray(axes, dtype=object).ravel():
        lines = []
        for line in ax.lines:
            lines.append((
                np.asarray(line.get_xdata(), dtype=float),
                np.asarray(line.get_ydata(), dtype=float),
            ))
        payload.append((ax.get_title(), lines))
    return payload


def test_Ba1_draw_profile_facet_vector_matches_typed_owner():
    """draw() and profile() must produce the same vector-facet result."""
    df = _asymmetric_vector_df()
    kwargs = dict(
        bins=4,
        group_by="g",
        facet_by="vector",
        stats=["mean"],
    )

    typed_fig, typed_axes, typed_stats = DFDraw(df).profile(
        "[y0,y1]:x", **kwargs
    )
    draw_fig, draw_axes, draw_stats = DFDraw(df).draw(
        "[y0,y1]:x", type="profile", **kwargs
    )

    try:
        assert typed_stats["faceted"] is True
        assert draw_stats["faceted"] is True
        assert typed_stats["facet_by"] == draw_stats["facet_by"] == "vector"
        assert typed_stats["groups"] == draw_stats["groups"] == ["y0", "y1"]
        assert typed_stats["n_groups"] == draw_stats["n_groups"] == 2

        typed_payload = _axis_line_payload(typed_axes)
        draw_payload = _axis_line_payload(draw_axes)
        assert [x[0] for x in typed_payload] == ["vector=y0", "vector=y1"]
        assert [x[0] for x in draw_payload] == ["vector=y0", "vector=y1"]
        assert len(typed_payload) == len(draw_payload) == 2

        for (typed_title, typed_lines), (draw_title, draw_lines) in zip(
            typed_payload, draw_payload
        ):
            assert typed_title == draw_title
            assert len(typed_lines) == len(draw_lines)
            for (tx, ty), (dx, dy) in zip(typed_lines, draw_lines):
                np.testing.assert_allclose(dx, tx, rtol=0, atol=0)
                np.testing.assert_allclose(dy, ty, rtol=0, atol=0)
    finally:
        plt.close(typed_fig)
        plt.close(draw_fig)


def test_Ba2_vector_facets_keep_expression_components_distinct():
    """Independent coarse oracle: y0 and y1 must occupy different facets."""
    df = _asymmetric_vector_df()
    fig, axes, stats = DFDraw(df).draw(
        "[y0,y1]:x",
        type="profile",
        bins=4,
        group_by="g",
        facet_by="vector",
    )
    try:
        axes = np.asarray(axes, dtype=object).ravel()
        assert len(axes) == 2
        assert [ax.get_title() for ax in axes] == ["vector=y0", "vector=y1"]
        assert stats["groups"] == ["y0", "y1"]

        # Deliberately asymmetric construction: every y1 value is above every
        # y0 value.  This catches component loss/swapping without depending on
        # dfdraw's own profile arithmetic.
        y0_rendered = np.concatenate([
            np.asarray(line.get_ydata(), dtype=float)
            for line in axes[0].lines
            if len(line.get_ydata())
        ])
        y1_rendered = np.concatenate([
            np.asarray(line.get_ydata(), dtype=float)
            for line in axes[1].lines
            if len(line.get_ydata())
        ])
        assert np.nanmax(y0_rendered) < np.nanmin(y1_rendered)
    finally:
        plt.close(fig)
