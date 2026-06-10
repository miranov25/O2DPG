"""Phase 13.54.DF — Gallery-found bug fixes.

Tests for the two dfdraw bugs surfaced by the ADF time_series_draw.py
real-data visual gallery (per AD-TS-DRAW-001):

BUG_dfdraw_20260609_scatter_auto_title:
    Gallery G1.04 — scatter with auto_title= crashed because draw_scatter()
    signature did not accept it. Phase 13.54.DF adds the parameter to
    draw_scatter(), DFDraw.scatter(), and explicitly forwards it through the
    faceted dispatch path (mirror of hist2d).

BUG_dfdraw_20260610_hist2d_time_format_epoch:
    Gallery G3.16 — hist2d with time_format= and float64 epoch-second
    timestamps (~1.776e9) crashed with OverflowError because the Phase 13.51
    S-8 conversion only handled datetime64, not float epoch-seconds.
    Phase 13.54.DF adds the elif-branch that mirrors draw_hist() at L440-453.

Six tests cover both fixes plus regression checks:

T1: DFDraw(df).scatter(auto_title=True) — direct call works
T2: adf.draw(type='scatter', auto_title=True) — dispatch path works
T3: scatter without auto_title — no regression (Phase 13.51 baseline)
T4: hist2d with float epoch + time_format — no overflow, HH:MM labels
T5: hist2d with datetime64 + time_format — Phase 13.51 S-8 regression check
T6: hist2d with non-time float + time_format=None — else-else branch regression

References:
- BUG_dfdraw_20260609_scatter_auto_title.md
- BUG_dfdraw_20260610_hist2d_time_format_epoch.md
- AD-TS-DRAW-001 (mandatory gallery validation as pre-tag gate)
- Sonnet65_GalleryBugReports_PanelSummary_20260610.md (10-reviewer panel)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from dfextensions.dfdraw import DFDraw


# =============================================================================
# Fixtures
# =============================================================================

def _make_basic_df(n=500, seed=42):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "x": rng.standard_normal(n),
        "y": rng.standard_normal(n),
        "g": rng.choice(["a", "b", "c"], n),
    })


def _make_epoch_df(n=2000, seed=42):
    """Mirror gallery G3.16 fixture: float64 epoch seconds near ALICE Run 3
    timescale (~1.776e9 s = 2026 calendar year). Pre-fix this would
    OverflowError in DateFormatter."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "time_s": np.linspace(1.7760e9, 1.7761e9, n),  # float64 epoch seconds
        "y": rng.standard_normal(n),
        "x": rng.standard_normal(n),
    })


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


# =============================================================================
# BUG_dfdraw_20260609_scatter_auto_title — fix tests
# =============================================================================

def test_T1_scatter_direct_auto_title_sets_title():
    """T1: DFDraw(df).scatter('y:x', auto_title=True) — direct (non-dispatch)
    call must produce a title via the _auto_title helpers and NOT raise
    PathCollection.set() AttributeError.

    Pre-Phase-13.54: AttributeError because draw_scatter() signature did
    not consume auto_title=, so it leaked into ax.scatter() kwargs.
    """
    df = _make_basic_df()
    fig, ax, _ = DFDraw(df).scatter("y:x", auto_title=True)
    title = ax.get_title()
    assert title and title.strip(), (
        f"scatter(auto_title=True) must produce a non-empty title; got {title!r}"
    )
    # Mirror histogram.py / profile.py auto_title content: includes y vs x names
    assert "y" in title and "x" in title, (
        f"auto_title content should reference x_name and y_name; got {title!r}"
    )


def test_T2_draw_type_scatter_auto_title_routes_through_dispatch():
    """T2: adf.draw('y:x', type='scatter', auto_title=True) — full dispatch
    path (draw → scatter → draw_scatter) must produce a title. This is the
    exact gallery G1.04 call pattern.

    Validates: (a) auto_title rides in via draw()'s **kwargs into
    DFDraw.scatter() where the new named param captures it, (b) it forwards
    to draw_scatter() via the R-2 pattern added in Phase 13.54.
    """
    df = _make_basic_df()
    fig, ax, _ = DFDraw(df).draw("y:x", type="scatter", auto_title=True)
    title = ax.get_title()
    assert title and title.strip(), (
        f"draw(type='scatter', auto_title=True) must set title; got {title!r}"
    )


def test_T3_scatter_no_auto_title_regression_baseline():
    """T3: scatter() without auto_title — must NOT produce a title (i.e.
    Phase 13.54 fix didn't accidentally make scatter always titled).

    Negative-equivalence to T1: prove the parameter actually controls the
    behavior rather than being a no-op."""
    df = _make_basic_df()
    fig, ax, _ = DFDraw(df).scatter("y:x")
    title = ax.get_title()
    assert title == "" or title is None, (
        f"scatter() without auto_title must leave title empty; got {title!r}"
    )

    # Companion check: explicit title= still wins over auto_title=True
    fig2, ax2, _ = DFDraw(df).scatter("y:x", title="explicit", auto_title=True)
    assert ax2.get_title() == "explicit", (
        f"explicit title= must win over auto_title=True; got {ax2.get_title()!r}"
    )


# =============================================================================
# BUG_dfdraw_20260610_hist2d_time_format_epoch — fix tests
# =============================================================================

def test_T4_hist2d_float_epoch_time_format_no_overflow():
    """T4: hist2d with float64 epoch-second column + time_format must NOT
    raise OverflowError, AND must produce HH:MM tick labels on x-axis.
    Mirror of gallery G3.16.

    Pre-Phase-13.54: OverflowError in DateFormatter because raw 1.776e9 was
    interpreted as ordinal days. Phase 13.54 adds the elif-branch that
    converts float epoch seconds via pd.to_datetime(unit='s') + mdates.date2num,
    mirroring draw_hist() at L440-453."""
    df = _make_epoch_df()
    # The call that crashed pre-fix
    fig, ax, _ = DFDraw(df).draw(
        "y:time_s", type="hist2d", bins=100, time_format="%H:%M",
    )
    # Tick labels should look like HH:MM (any digit + colon + digits)
    labels = [t.get_text() for t in ax.get_xticklabels()]
    nonempty = [l for l in labels if l.strip()]
    assert nonempty, "x-axis tick labels must not be empty"
    assert any(":" in l for l in nonempty), (
        f"hist2d + time_format='%H:%M' should produce HH:MM labels; got {nonempty}"
    )
    # xlim should be in ordinal-day range (not raw 1e9 seconds) — proves
    # the date2num conversion fired rather than the raw .astype(float) path.
    xlo, xhi = ax.get_xlim()
    assert xlo < 1e6, (
        f"xlim post-conversion should be ordinal days (~20000), not raw "
        f"seconds (~1e9); got {xlo}"
    )


def test_T5_hist2d_datetime64_time_format_phase_13_51_regression():
    """T5: hist2d with datetime64 + time_format — Phase 13.51 S-8 path must
    still work after the Phase 13.54 elif-branch addition.

    Regression check: ensure my v1.5.4 fix didn't break the existing
    datetime64 dtype detection path."""
    df = _make_epoch_df()
    df["time_dt"] = pd.to_datetime(df["time_s"], unit="s")  # datetime64[ns]
    fig, ax, _ = DFDraw(df).draw(
        "y:time_dt", type="hist2d", bins=100, time_format="%H:%M",
    )
    labels = [t.get_text() for t in ax.get_xticklabels()]
    nonempty = [l for l in labels if l.strip()]
    assert any(":" in l for l in nonempty), (
        f"datetime64 path should still produce HH:MM labels (Phase 13.51 S-8); "
        f"got {nonempty}"
    )


def test_T6_hist2d_non_time_float_no_time_format_else_else_branch():
    """T6: hist2d with plain float columns and time_format=None — the
    "else-else" branch must still convert via raw .astype(float) without
    invoking mdates.date2num.

    Regression check: prove the Phase 13.54 elif doesn't pollute the no-time
    code path (which is the vast majority of hist2d usage)."""
    df = _make_basic_df(n=1000)
    fig, ax, _ = DFDraw(df).draw("y:x", type="hist2d", bins=30)
    # Default x-range should reflect raw x (gaussian, ~[-3, 3]), not
    # ordinal-day range
    xlo, xhi = ax.get_xlim()
    assert -10 < xlo < 0 and 0 < xhi < 10, (
        f"non-time float path should yield gaussian-range xlim; got ({xlo}, {xhi})"
    )

    # Companion check: y-axis float-epoch symmetry (the new elif applies to
    # both x_data and y_data conversion sites in hist2d — verify y-axis path)
    df_yepoch = pd.DataFrame({
        "x": np.random.default_rng(7).standard_normal(2000),
        "time_s": np.linspace(1.7760e9, 1.7761e9, 2000),
    })
    fig2, ax2, _ = DFDraw(df_yepoch).draw(
        "time_s:x", type="hist2d", bins=50, time_format="%H:%M",
    )
    # The y-axis data was epoch-seconds; post-conversion ylim should be in
    # ordinal-days range (~20000s), NOT raw 1e9. The Phase 13.51 design only
    # applies the DateFormatter to x-axis; y-axis y_data was already
    # converted but tick labels remain numeric — we only check the data
    # space, not the formatter.
    ylo, yhi = ax2.get_ylim()
    assert ylo < 1e6, (
        f"y-axis float-epoch should also convert to ordinal days; got ylim "
        f"({ylo}, {yhi}). This verifies Phase 13.54's symmetric x+y fix."
    )
