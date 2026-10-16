"""PHASE_13_86 Stage C-a: width-aware auto-title selection fitting."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dfdraw import DFDraw
from dfdraw.plots._auto_title import build_auto_title, apply_auto_title


LONG_SELECTION = (
    "(abs(b)<0.95)&(abs(c)<0.95)&(a>-2.9)&(a<2.9)&"
    "(abs(b-sin(a))<0.3)&(abs(c-cos(a))<0.3)"
)


def _subtitle_text(ax):
    subs = [t.get_text() for t in ax.texts if getattr(t, "_is_auto_subtitle", False)]
    assert len(subs) == 1
    return subs[0]


def test_CA1_wide_figure_keeps_full_selection_but_narrow_figure_elides():
    """Same selection must fit differently as figure width changes."""
    td = build_auto_title("a", "b", selection=LONG_SELECTION, parts={"expr", "sel"})

    fig_w, ax_w = plt.subplots(figsize=(14, 3))
    fig_n, ax_n = plt.subplots(figsize=(3.2, 3))
    try:
        apply_auto_title(ax_w, td)
        apply_auto_title(ax_n, td)
        wide = _subtitle_text(ax_w)
        narrow = _subtitle_text(ax_n)

        assert wide == LONG_SELECTION
        assert narrow.endswith("...")
        assert len(narrow) < len(wide)
    finally:
        plt.close(fig_w)
        plt.close(fig_n)


def test_CA2_build_auto_title_preserves_selection_until_render_width_is_known():
    """The semantic title builder must no longer apply a fixed character cut."""
    td = build_auto_title("a", "b", selection=LONG_SELECTION, parts={"expr", "sel"})
    assert td["sub"] == LONG_SELECTION


def test_CA3_faceted_wide_figure_keeps_full_selection_in_figure_title():
    """Production bug: a wide multi-panel figure must not inherit a 72-char cut."""
    rng = np.random.default_rng(1)
    n = 4000
    a = rng.uniform(-3, 3, n)
    df = pd.DataFrame({
        "a": a,
        "b": np.sin(a) + 0.1 * rng.normal(size=n),
        "c": np.cos(a) + 0.1 * rng.normal(size=n),
        "k": rng.integers(0, 3, n),
    })
    fig, _, _ = DFDraw(df).draw(
        "b:a", type="scatter", selection=LONG_SELECTION,
        facet_by="k", auto_title=True,
    )
    try:
        lines = fig.get_suptitle().split("\n")
        assert len(lines) >= 2
        assert lines[1] == LONG_SELECTION
    finally:
        plt.close(fig)
