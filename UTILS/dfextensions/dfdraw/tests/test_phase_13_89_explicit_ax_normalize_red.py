"""PHASE_13_89_DF — explicit-axis embedding RED/GREEN contract."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from dfextensions.dfdraw import DFDraw


def _df():
    n = 1200
    x = np.linspace(-1.0, 1.0, n)
    y = 5.0 + 0.4*x + 0.05*np.sin(np.arange(n))
    return pd.DataFrame({
        "x": x,
        "y": y,
        "y2": 2.0*y,
        "sel": np.where(np.arange(n) % 2 == 0, -1, 1),
        "g": np.where(np.arange(n) % 3 == 0, "A", "B"),
        "facet": np.where(x < 0.0, "L", "R"),
    })


def _figset():
    return set(plt.get_fignums())


def _plain(df, normalize, layout, ax=None, door="typed", **kwargs):
    call = dict(
        bins=16,
        range=(-1.0, 1.0),
        normalize=normalize,
        normalize_layout=layout,
        return_data=True,
        **kwargs,
    )
    if ax is not None:
        call["ax"] = ax
    d = DFDraw(df)
    if door == "typed":
        return d.profile("[y,y2]:x", **call)
    return d.draw("[y,y2]:x", type="profile", **call)


def _assert_norm_equal(lhs, rhs):
    pd.testing.assert_frame_equal(
        lhs["normalize_data"].reset_index(drop=True),
        rhs["normalize_data"].reset_index(drop=True),
        check_dtype=True,
        check_exact=True,
    )


@pytest.mark.parametrize("door", ["typed", "draw"])
@pytest.mark.parametrize("normalize", ["delta", "ratio"])
def test_R1_R3_plain_overlay_diff_embeds_in_caller_slot_and_preserves_numbers(door, normalize):
    df = _df()
    ref_fig, _, ref_stats = _plain(df, normalize, "overlay+diff", door=door)
    caller_fig, placeholder = plt.subplots(figsize=(7.5, 5.5))
    caller_fig.suptitle("CALLER SUPTITLE")
    size0 = caller_fig.get_size_inches().copy()
    before = _figset()
    try:
        fig, ax_top, stats = _plain(
            df, normalize, "overlay+diff", ax=placeholder, door=door,
            auto_title=True, figsize=(20.0, 20.0),
        )
        assert fig is caller_fig
        assert ax_top.figure is caller_fig
        assert stats["ax_diff"].figure is caller_fig
        assert placeholder not in caller_fig.axes
        assert caller_fig._suptitle.get_text() == "CALLER SUPTITLE"
        np.testing.assert_allclose(caller_fig.get_size_inches(), size0, rtol=0, atol=0)
        assert _figset() == before
        _assert_norm_equal(stats, ref_stats)
    finally:
        plt.close(ref_fig)
        plt.close(caller_fig)


def test_R2_draw_selection_vector_production_shape_embeds_and_preserves_numbers():
    df = _df()
    kwargs = dict(
        type="profile", bins=16, range=(-1.0, 1.0),
        selection_vector=["sel<0", "sel>=0"], vector_compose="outer",
        normalize="delta", normalize_layout="overlay+diff", return_data=True,
    )
    ref_fig, _, ref_stats = DFDraw(df).draw("y:x", **kwargs)
    caller_fig, placeholder = plt.subplots()
    before = _figset()
    try:
        fig, ax_top, stats = DFDraw(df).draw("y:x", ax=placeholder, **kwargs)
        assert fig is caller_fig
        assert ax_top.figure is caller_fig
        assert stats["ax_diff"].figure is caller_fig
        assert placeholder not in caller_fig.axes
        assert _figset() == before
        _assert_norm_equal(stats, ref_stats)
    finally:
        plt.close(ref_fig)
        plt.close(caller_fig)


@pytest.mark.parametrize("normalize", ["delta", "ratio"])
def test_R4_diff_only_uses_exact_supplied_axis(normalize):
    df = _df()
    ref_fig, _, ref_stats = _plain(df, normalize, "diff_only")
    caller_fig, supplied_ax = plt.subplots()
    caller_fig.suptitle("KEEP ME")
    size0 = caller_fig.get_size_inches().copy()
    before = _figset()
    try:
        fig, returned_ax, stats = _plain(
            df, normalize, "diff_only", ax=supplied_ax,
            auto_title=True, figsize=(20.0, 20.0),
        )
        assert fig is caller_fig
        assert returned_ax is supplied_ax
        assert stats["ax_diff"] is supplied_ax
        assert len(supplied_ax.lines) > 0
        assert caller_fig._suptitle.get_text() == "KEEP ME"
        np.testing.assert_allclose(caller_fig.get_size_inches(), size0, rtol=0, atol=0)
        assert _figset() == before
        _assert_norm_equal(stats, ref_stats)
    finally:
        plt.close(ref_fig)
        plt.close(caller_fig)


def test_R5_multisubplot_embedding_changes_only_target_slot():
    df = _df()
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.0))
    target = axes[0, 0]
    siblings = [axes[0, 1], axes[1, 0], axes[1, 1]]
    fig.suptitle("OWNER TITLE")
    for i, ax in enumerate(siblings):
        ax.plot([0, 1], [i, i + 1])
    ids0 = [id(ax) for ax in siblings]
    pos0 = [ax.get_position().frozen() for ax in siblings]
    lines0 = [len(ax.lines) for ax in siblings]
    size0 = fig.get_size_inches().copy()
    before = _figset()
    try:
        out_fig, ax_top, stats = DFDraw(df).profile(
            "[y,y2]:x", bins=16, range=(-1, 1), normalize="delta",
            normalize_layout="overlay+diff", ax=target, auto_title=True,
            return_data=True,
        )
        assert out_fig is fig
        assert target not in fig.axes
        assert ax_top in fig.axes and stats["ax_diff"] in fig.axes
        assert [id(ax) for ax in siblings] == ids0
        for ax, p0, n0 in zip(siblings, pos0, lines0):
            p = ax.get_position()
            np.testing.assert_allclose([p.x0,p.y0,p.x1,p.y1],[p0.x0,p0.y0,p0.x1,p0.y1], rtol=0, atol=1e-12)
            assert len(ax.lines) == n0
        assert fig._suptitle.get_text() == "OWNER TITLE"
        np.testing.assert_allclose(fig.get_size_inches(), size0, rtol=0, atol=0)
        assert _figset() == before
    finally:
        plt.close(fig)


def test_R6_custom_add_axes_refuses_before_mutation_or_leak():
    df = _df()
    fig = plt.figure()
    ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
    axes0 = tuple(fig.axes)
    before = _figset()
    try:
        with pytest.raises((ValueError, NotImplementedError), match="ax|Axes|subplot|embed"):
            DFDraw(df).profile("[y,y2]:x", bins=16, range=(-1,1), normalize="delta", normalize_layout="overlay+diff", ax=ax)
        assert tuple(fig.axes) == axes0
        assert _figset() == before
    finally:
        plt.close(fig)


def test_R7_nonempty_placeholder_refuses_before_mutation_or_leak():
    df = _df()
    fig, ax = plt.subplots()
    line, = ax.plot([0,1],[0,1])
    axes0 = tuple(fig.axes)
    before = _figset()
    try:
        with pytest.raises((ValueError, NotImplementedError), match="ax|Axes|empty|embed|replace"):
            DFDraw(df).profile("[y,y2]:x", bins=16, range=(-1,1), normalize="delta", normalize_layout="overlay+diff", ax=ax)
        assert tuple(fig.axes) == axes0
        assert line in ax.lines
        assert _figset() == before
    finally:
        plt.close(fig)


@pytest.mark.parametrize("extra", [{"group_by":"g"},{"facet_by":"facet"}], ids=["grouped","faceted"])
def test_R8_grouped_or_faceted_normalize_with_ax_refuses_without_leak(extra):
    df = _df()
    fig, ax = plt.subplots()
    axes0 = tuple(fig.axes)
    before = _figset()
    try:
        with pytest.raises((ValueError, NotImplementedError), match="ax|Axes|group|facet|embed|support"):
            DFDraw(df).profile("[y,y2]:x", bins=16, range=(-1,1), normalize="delta", normalize_layout="overlay+diff", ax=ax, **extra)
        assert tuple(fig.axes) == axes0
        assert _figset() == before
    finally:
        plt.close(fig)


@pytest.mark.parametrize("door", ["typed", "draw"])
def test_R9_hist_normalize_ax_preserves_existing_loud_refusal(door):
    df = _df()
    fig, ax = plt.subplots()
    axes0 = tuple(fig.axes)
    before = _figset()
    try:
        kwargs = dict(bins=16, range=(0,15), normalize="ratio", normalize_layout="overlay+diff", ax=ax)
        with pytest.raises((ValueError, NotImplementedError), match="ax|Axes|support"):
            if door == "typed":
                DFDraw(df).hist("[y,y2]", **kwargs)
            else:
                DFDraw(df).draw("[y,y2]", type="hist", **kwargs)
        assert tuple(fig.axes) == axes0
        assert _figset() == before
    finally:
        plt.close(fig)


def test_R10_profile_and_hist_standalone_controls_remain_supported():
    df = _df()
    pfig, _, pstats = DFDraw(df).profile("[y,y2]:x", bins=16, range=(-1,1), normalize="ratio", normalize_layout="overlay+diff", return_data=True)
    hfig, _, hstats = DFDraw(df).hist("[y,y2]", bins=16, range=(0,15), normalize="ratio", normalize_layout="overlay+diff")
    try:
        assert "normalize_data" in pstats
        assert "normalize_data" in hstats
        assert len(pfig.axes) >= 2
        assert len(hfig.axes) >= 2
    finally:
        plt.close(pfig)
        plt.close(hfig)


def test_R11_ax_plus_save_refuses_without_mutating_caller(tmp_path):
    df = _df()
    fig, ax = plt.subplots()
    outfile = tmp_path / "must_not_be_saved.png"
    axes0 = tuple(fig.axes)
    before = _figset()
    try:
        with pytest.raises((ValueError, NotImplementedError), match="save|ax|Axes|caller|owner"):
            DFDraw(df).profile("[y,y2]:x", bins=16, range=(-1,1), normalize="delta", normalize_layout="overlay+diff", ax=ax, save=str(outfile))
        assert tuple(fig.axes) == axes0
        assert not outfile.exists()
        assert _figset() == before
    finally:
        plt.close(fig)
