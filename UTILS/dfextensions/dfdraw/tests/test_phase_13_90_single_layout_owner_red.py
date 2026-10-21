"""PHASE_13_90_DF — single-owner layout contract for faceted rendering.

RED-first deterministic proof for the bounded performance fix:
each completed faceted figure gets exactly one whole-figure tight_layout()
pass, owned by the facet coordinator. Standalone non-faceted plots keep
their existing one-pass behavior.

The suite deliberately uses call counts, not wall-clock thresholds.
"""

from __future__ import annotations

from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from dfextensions.dfdraw import DFDraw


def _df(n=900):
    i = np.arange(n)
    x = np.linspace(-3.0, 3.0, n)
    y = 0.5 * x + np.sin(i / 17.0)
    z = 0.25 * x - np.cos(i / 23.0)
    facet = np.linspace(0.0, 9.0, n, endpoint=False)
    return pd.DataFrame({
        "x": x,
        "y": y,
        "z": z,
        "a": y + 10.0,
        "b": z + 20.0,
        "c": y - z + 30.0,
        "d": 2.0 * y + z + 40.0,
        "facet": facet,
    })


def _count_tight_layout(call):
    original = plt.tight_layout
    with patch.object(plt, "tight_layout", wraps=original) as mocked:
        fig, axes, stats = call()
    return mocked.call_count, fig, axes, stats


@pytest.mark.parametrize(
    "kind,expr",
    [
        ("profile", "y:x"),
        ("hist", "y"),
        ("scatter", "y:x"),
        ("hist2d", "y:x"),
    ],
)
def test_R1_faceted_plot_has_exactly_one_final_layout_pass(kind, expr):
    """Binding 90-C1: all shared facet kinds use one final layout owner."""
    df = _df()
    d = DFDraw(df)

    def render():
        fn = getattr(d, kind)
        return fn(
            expr,
            facet_by="facet",
            facet_by_bins=9,
            ncols=3,
        )

    count, fig, axes, stats = _count_tight_layout(render)
    try:
        assert count == 1, (
            f"{kind} faceted rendering must perform exactly one whole-figure "
            f"tight_layout pass; observed {count}"
        )
        assert stats is not None
    finally:
        plt.close(fig)


def test_R2_new_phase_13_91_hist_vector_facets_have_one_layout_pass():
    """13.91 follow-on requirement: new hist-vector facet path is covered."""
    df = _df()
    d = DFDraw(df)

    def render():
        return d.hist(
            "[a,b,c,d]",
            facet_by="vector",
            bins=12,
            ncols=2,
            auto_title=False,
        )

    count, fig, axes, stats = _count_tight_layout(render)
    try:
        assert count == 1, (
            "PHASE_13_91 histogram vector facets must inherit the facet "
            f"single-owner layout contract; observed {count} layout passes"
        )
        assert stats["n_groups"] == 4
        assert stats["groups"] == ["a", "b", "c", "d"]
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "kind,expr",
    [
        ("profile", "y:x"),
        ("hist", "y"),
        ("scatter", "y:x"),
        ("hist2d", "y:x"),
    ],
)
def test_R3_standalone_nonfaceted_layout_behavior_is_unchanged(kind, expr):
    """Non-faceted owners still perform their normal one layout pass."""
    df = _df(300)
    d = DFDraw(df)

    def render():
        return getattr(d, kind)(expr)

    count, fig, axes, stats = _count_tight_layout(render)
    try:
        assert count == 1, (
            f"standalone {kind} must retain its existing one-layout behavior; "
            f"observed {count}"
        )
        assert stats is not None
    finally:
        plt.close(fig)
