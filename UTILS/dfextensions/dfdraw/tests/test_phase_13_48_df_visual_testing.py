"""
Phase 13.48.DF — Automated Visual Testing (Tier 1: primitive-only)

Renderer-free, deterministic, backend-independent visual checks on OPEN figures
(before plt.close()). Origin: audit PR-3 (savefig call sites, no figure rendered).

Design (proposal v1.4, panel-approved 4×[OK]+2×[!], 0×[X]):
  - VisualCheck(fig, stats, df): collect-all-then-assert (full defect list).
  - Cell iteration over fig.axes, NOT stats[(row,col)] (P1, Claude48/Opus1 —
    faceted stats is flat; fig.axes is dispatcher-agnostic).
  - Two cell sets (P2, Opus1 — avoids the V.2 tautology AND the ragged-grid
    padding false-positive Claude48 found in v1.4):
      * visible_cell_axes = full grid MINUS hidden padding (set_visible(False))
        → V.2 (empty-cell), V.5 (grid), V.6 (shared-axis), V.9 (title dedup)
      * data_cell_axes    = visible cells that actually drew data
        → V.3/V.4/V.7/V.8 (counts/colors need data)
    A visible cell with no data artist IS the dropped-data defect V.2 catches.
  - Cell→facet-value via zip(sorted unique facet values) — dtype-safe; avoids
    the title-string parse (CRR §2 / Claude48 P3: 'sec=0' str vs int 0 column).
  - Counts use ax.containers (errorbar series), NOT len(ax.lines) (inflated by
    errorbar cap lines: 9 lines for 3 groups). Distinctness = exact distinct
    RGBA count (no HSV tolerance). Bounds use get_xydata()/get_offsets() only
    (errorbar caps legitimately extend ±sigma).

All mechanics above were source-verified against the PHASE_13_46_DF_FIX1_END
package before this suite was written.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from dfextensions.dfdraw import DFDraw
from dfextensions.dfdraw.drawer import _get_suptitle  # Phase 13.46 C-4 helper


# --------------------------------------------------------------------------- #
# Primitive helpers (callable standalone — Sonet51 minority view)
# --------------------------------------------------------------------------- #

def grid_geometry(fig):
    """(nrows, ncols) of the facet grid from gridspec, or (1, 1)."""
    if not fig.axes:
        return (1, 1)
    ss = fig.axes[0].get_subplotspec()
    if ss is None:
        return (1, 1)
    return ss.get_gridspec().get_geometry()


def grid_cell_axes(fig):
    """The nrows*ncols subplot axes in the grid (includes hidden padding)."""
    nrows, ncols = grid_geometry(fig)
    return list(fig.axes[: nrows * ncols])


def visible_cell_axes(fig):
    """Grid cells the user is meant to see — hidden padding excluded.

    dfdraw hides ragged-grid padding via set_visible(False); excluding it makes
    the empty-cell check able to FAIL on a genuinely-dropped cell without
    false-failing on legitimate padding (Claude48 v1.4 P1)."""
    return [ax for ax in grid_cell_axes(fig) if ax.get_visible()]


def data_cell_axes(fig):
    """Visible grid cells that actually drew data artists."""
    return [ax for ax in visible_cell_axes(fig) if ax.lines or ax.collections]


def points_in_bounds(ax):
    """All scatter offsets / line vertices lie within the axes' xlim x ylim."""
    (xlo, xhi), (ylo, yhi) = ax.get_xlim(), ax.get_ylim()
    pts = []
    for coll in ax.collections:
        off = coll.get_offsets()
        if len(off):
            pts.append(np.asarray(off))
    for line in ax.lines:
        xy = line.get_xydata()
        if len(xy):
            pts.append(np.asarray(xy))
    if not pts:
        return True
    p = np.vstack(pts)
    eps = 1e-9
    return bool(np.all((p[:, 0] >= xlo - eps) & (p[:, 0] <= xhi + eps) &
                       (p[:, 1] >= ylo - eps) & (p[:, 1] <= yhi + eps)))


def series_count(ax):
    """Number of data series in a (possibly grouped) profile/scatter cell.

    Uses ax.containers (one errorbar container per group) — NOT len(ax.lines),
    which counts errorbar cap lines too (9 lines for 3 groups)."""
    if ax.containers:
        return len(ax.containers)
    if ax.collections:
        return len(ax.collections)
    return len(ax.lines)


def distinct_colors(ax):
    """Exact count of distinct RGBA colors among the cell's data artists."""
    seen = set()
    for line in ax.lines:
        c = line.get_color()
        seen.add(c if isinstance(c, str) else tuple(np.round(np.asarray(c), 4)))
    for patch in ax.patches:
        seen.add(tuple(np.round(np.asarray(patch.get_facecolor()), 4)))
    for coll in ax.collections:
        fc = coll.get_facecolors()
        for row in np.atleast_2d(fc):
            seen.add(tuple(np.round(np.asarray(row), 4)))
    return len(seen)


def legend_entry_count(ax):
    leg = ax.get_legend()
    return len(leg.get_texts()) if leg is not None else 0


def colorbar_present(fig):
    """Heuristic figure-axis scan — self-validating (no stats flag exists)."""
    return any(getattr(ax, "_colorbar", None) is not None or
               (ax.get_label() == "<colorbar>") for ax in fig.axes)


def shared_axis_consistent(axes, which="x"):
    """All given cells share one x (or y) range."""
    lims = {ax.get_xlim() if which == "x" else ax.get_ylim() for ax in axes}
    return len(lims) == 1


def title_not_duplicated(fig, cell_axes):
    """Suptitle is populated and the auto-title is not repeated per-cell.

    Per-cell facet bin labels (e.g. 'sec=0') ARE expected and not checked."""
    sup = _get_suptitle(fig)
    if not sup:
        return False
    return all(sup not in ax.get_title() for ax in cell_axes)


def title_content(fig, must_contain):
    sup = _get_suptitle(fig) or ""
    return all(tok in sup for tok in must_contain)


# --------------------------------------------------------------------------- #
# VisualCheck — collect-all-then-assert
# --------------------------------------------------------------------------- #

class VisualCheck:
    def __init__(self, fig, stats=None, df=None):
        self.fig = fig
        self.stats = stats
        self.df = df
        self.defects = []

    def fail(self, name, detail):
        self.defects.append(f"{name}: {detail}")

    def check_bounds(self):
        for i, ax in enumerate(data_cell_axes(self.fig) or self.fig.axes):
            if not points_in_bounds(ax):
                self.fail("bounds", f"cell {i} has data outside xlim x ylim")
        return self

    def check_cells_nonempty(self):
        for i, ax in enumerate(visible_cell_axes(self.fig)):
            if not (ax.lines or ax.collections):
                self.fail("empty_cell", f"visible cell {i} drew no data artist")
        return self

    def check_grid(self, expected_shape=None, expected_visible=None):
        shape = grid_geometry(self.fig)
        if expected_shape is not None and shape != expected_shape:
            self.fail("grid", f"geometry {shape} != expected {expected_shape}")
        if expected_visible is not None and len(visible_cell_axes(self.fig)) != expected_visible:
            self.fail("grid", f"visible cells {len(visible_cell_axes(self.fig))} != {expected_visible}")
        return self

    def check_counts(self, facet_col, group_col, selection=None, kind="series"):
        cells = data_cell_axes(self.fig)
        facet_vals = sorted(self.df[facet_col].unique())
        for ax, val in zip(cells, facet_vals):          # dtype-safe mapping
            dfc = self.df[self.df[facet_col] == val]
            if selection:
                dfc = dfc.query(selection)
            n_expected = dfc[group_col].nunique()
            got = legend_entry_count(ax) if kind == "legend" else series_count(ax)
            if got != n_expected:
                self.fail("count", f"cell {val}: {kind}={got} != n_groups={n_expected}")
        return self

    def check_colors(self, n_groups, palette_size=10):
        for ax, val in zip(data_cell_axes(self.fig), range(99)):
            n = distinct_colors(ax)
            if n < min(n_groups, palette_size):
                self.fail("colors", f"cell {val}: {n} distinct < min({n_groups},{palette_size})")
        return self

    def check_shared_axis(self, which="x"):
        cells = visible_cell_axes(self.fig)
        if len(cells) > 1 and not shared_axis_consistent(cells, which):
            self.fail("shared_axis", f"{which}lims differ across cells when share active")
        return self

    def check_title_dedup(self):
        if not title_not_duplicated(self.fig, visible_cell_axes(self.fig)):
            self.fail("title_dedup", "suptitle empty or duplicated per-cell")
        return self

    def check_title_content(self, must_contain):
        if not title_content(self.fig, must_contain):
            self.fail("title_content", f"suptitle missing {must_contain}")
        return self

    def assert_clean(self):
        if self.defects:
            raise AssertionError("VisualCheck defects:\n  - " + "\n  - ".join(self.defects))


# --------------------------------------------------------------------------- #
# Fixtures — three figures (N-2 precondition: discrete group_col, >=min/cell)
# --------------------------------------------------------------------------- #

@pytest.fixture
def df_basic():
    rs = np.random.RandomState(0)
    return pd.DataFrame({
        "x": rs.normal(0, 1, 1200),
        "y": rs.normal(0, 1, 1200),
        "g": rs.randint(0, 3, 1200),      # discrete group_col (3 groups)
        "sec": rs.randint(0, 2, 1200),    # facet col — fills a 1x2 grid (no padding)
    })


@pytest.fixture
def df_ragged():
    """5 facet values -> ragged 2x3 grid (1 hidden padding cell) for V.2 padding-safety."""
    rs = np.random.RandomState(1)
    return pd.DataFrame({
        "x": rs.normal(0, 1, 2000),
        "y": rs.normal(0, 1, 2000),
        "g": rs.randint(0, 3, 2000),
        "sec": rs.randint(0, 5, 2000),
    })


# --------------------------------------------------------------------------- #
# Tests V.1 - V.10
# --------------------------------------------------------------------------- #

class TestPhase1348VisualPrimitive:

    # -- Fig 1: scatter range="minmax" -------------------------------------- #

    def test_v1_scatter_points_in_bounds(self, df_basic):
        """V.1 — C-9 regression lock: every plotted point lies within limits."""
        fig, ax, stats = DFDraw(df_basic).scatter("y:x", range="minmax")
        VisualCheck(fig, stats, df_basic).check_bounds().assert_clean()
        plt.close(fig)

    # -- Fig 2: faceted grouped profile ------------------------------------- #

    def test_v2_every_visible_cell_nonempty(self, df_basic):
        """V.2 — visible (non-padding) grid cells each drew data."""
        fig, axes, stats = DFDraw(df_basic).profile("y:x", bins=8, facet_by="sec", group_by="g")
        VisualCheck(fig, stats, df_basic).check_cells_nonempty().assert_clean()
        plt.close(fig)

    def test_v2_padding_safe_on_ragged_grid(self, df_ragged):
        """V.2 padding-safety (Claude48 v1.4 P1): a ragged 2x3 grid has a hidden
        padding cell; it must be EXCLUDED (not false-fail), and the check must
        still be able to fail (visible cells == facet values present)."""
        fig, axes, stats = DFDraw(df_ragged).profile("y:x", bins=8, facet_by="sec", group_by="g")
        nrows, ncols = grid_geometry(fig)
        assert nrows * ncols > df_ragged["sec"].nunique(), "fixture must produce padding"
        assert len(grid_cell_axes(fig)) > len(visible_cell_axes(fig)), "padding must be hidden"
        assert len(visible_cell_axes(fig)) == df_ragged["sec"].nunique()
        VisualCheck(fig, stats, df_ragged).check_cells_nonempty().assert_clean()  # padding excluded → clean
        plt.close(fig)

    def test_v3_artist_count_matches_groups(self, df_basic):
        """V.3 — per-cell data-series count == per-cell filtered n_groups."""
        fig, axes, stats = DFDraw(df_basic).profile("y:x", bins=8, facet_by="sec", group_by="g")
        VisualCheck(fig, stats, df_basic).check_counts("sec", "g", kind="series").assert_clean()
        plt.close(fig)

    def test_v4_per_group_colors_distinct(self, df_basic):
        """V.4 — AD-37: per-group colors are distinct (no color-cycle reset)."""
        fig, axes, stats = DFDraw(df_basic).profile("y:x", bins=8, facet_by="sec", group_by="g")
        VisualCheck(fig, stats, df_basic).check_colors(n_groups=3).assert_clean()
        plt.close(fig)

    def test_v5_facet_grid_shape(self, df_basic):
        """V.5 — grid geometry matches expected (figure-derived, not stats)."""
        fig, axes, stats = DFDraw(df_basic).profile("y:x", bins=8, facet_by="sec", group_by="g")
        VisualCheck(fig, stats, df_basic).check_grid(
            expected_visible=df_basic["sec"].nunique()).assert_clean()
        plt.close(fig)

    def test_v6_shared_axis_consistency(self, df_basic):
        """V.6 — faceted cells share one x-range (default shared axes)."""
        fig, axes, stats = DFDraw(df_basic).profile("y:x", bins=8, facet_by="sec", group_by="g")
        VisualCheck(fig, stats, df_basic).check_shared_axis("x").assert_clean()
        plt.close(fig)

    def test_v9_auto_title_not_duplicated(self, df_basic):
        """V.9 — C-3 lock: suptitle populated; auto-title not duplicated per-cell
        (facet bin labels per-cell ARE expected)."""
        fig, axes, stats = DFDraw(df_basic).profile(
            "y:x", bins=8, facet_by="sec", group_by="g", auto_title=True)
        VisualCheck(fig, stats, df_basic).check_title_dedup().assert_clean()
        plt.close(fig)

    def test_v10_title_content(self, df_basic):
        """V.10 — I-4: suptitle carries the plotted expression."""
        fig, axes, stats = DFDraw(df_basic).profile(
            "y:x", bins=8, facet_by="sec", group_by="g", auto_title=True)
        VisualCheck(fig, stats, df_basic).check_title_content(["x"]).assert_clean()
        plt.close(fig)

    # -- Fig 3: hist + group_by (legend) ------------------------------------ #

    def test_v7_legend_entries_match_groups(self, df_basic):
        """V.7 — legend entries == n_groups (per-cell filtered df oracle)."""
        fig, ax, stats = DFDraw(df_basic).hist("x", group_by="g", bins=20)
        leg = legend_entry_count(ax)
        assert leg == df_basic["g"].nunique(), f"legend {leg} != {df_basic['g'].nunique()}"
        plt.close(fig)

    def test_v8_color_cycle_distinct(self, df_basic):
        """V.8 — hist group_by colors distinct (color-cycle not reset)."""
        fig, ax, stats = DFDraw(df_basic).hist("x", group_by="g", bins=20)
        assert distinct_colors(ax) >= df_basic["g"].nunique()
        plt.close(fig)
